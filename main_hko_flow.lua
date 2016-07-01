unpack = unpack or table.unpack

require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
-- require 'ConvLSTM'
require 'utils'
require 'cunn'
require 'cutorch'
require 'display_flow'
require 'optim'

cutorch.setDevice(5)
-- init configure
paths.dofile('opts_hko.lua')

-- init log
local log = loadfile('log.lua')()
log.outfile = 'train_log'
log.level = opt.trainLogLevel or "trace"
log.info('[init] log level: ', log.level, ' output to file ', log.outfile)
startTrainUtils()
print(opt)
assert(opt.init)

rmspropconf = {learningRate = eta}

-- init model
log.info('[init] load model def from ', opt.modelFile)
dofile(opt.modelFile) 

if opt.continueFromPara then
    log.info(string.format("[init] ==> evaluating hko flow model: %s ", opt.modelPara))
    log.info(string.format("[init] ==> loading configure for rmsprop %s ", opt.contRmsConf))
    local para = torch.load(opt.modelPara)
    local p2 = para:cuda()
    log.info('[init] model: ', model)
    local p, g = model:getParameters()
    p:fill(1):cmul(p2)
    -- local p2, g2 = modelLoad:parameters()
    -- for i = 1, #p do
    --     p[i]:fill(1):cmul(p2[i]:cuda())
    -- end
    -- modelLoad = {}
    p2 = {}
    -- g2 = {}
    rmspropconf = torch.load(opt.contRmsConf)
else
    opt.contIter = 1
    -- >>>>>>
    -- init LSTM parameters to small values, uniformly distributed
    local lstm_params, lstm_grads = lstm:getParameters()
    lstm_params:uniform(-0.08, 0.08)
    -- init LSTM biases to (forget_bias, other_bias)
    lstm:initBias(0,0)
    -- call LSTM forget to reset the memory
    lstm:forget()
    -- <<<<<<<
end

-- init criterion
local criterion = nn.MSECriterion()
if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end

local err = 0

-- start train
local function main()
  log.trace("load opti done@")
  paths.dofile('data_hko.lua')
  log.trace("load data_hko done@")

  torch.manualSeed(opt.seed)
  log.info('[init] set seed ', opt.seed)
  datasetSeq = getdataSeqHko('train') -- we sample nSeq consecutive frames
  -- log.info('[init] Loaded ' .. datasetSeq:size() .. ' images')
  log.info('[init] ==> training model')
  
  -- TODO: init parameters of model 
  local epoch = 0
  local eta = opt.lr
  local loss = 0

  for iter = opt.contIter, opt.maxIter do
      -- define eval closure
      model:zeroGradParameters()
      lstm:forget() -- ConvLSTM forget
      local sample = datasetSeq[iter]
      local data = sample[1]  
      if opt.useGpu then
          data = data:cuda()
      end

      local inputTable = {}
      local inputEncTable = {}
      local opticalFlow
      local imflow
      local flowTable = {}
      local outputTable = {}
      local targetTable = {}

      for i = 1, opt.inputSeqLen - 1 do 
          --assert(lstm.cell == nil, tostring(lstm.cell))
          log.trace('[branch_memory] input data for encoder', i)
          local framesMemory = data[{{}, {i}, {}, {}, {}}]:select(2, 1)
          model:updateOutput(framesMemory)
          table.insert(inputTable, framesMemory)

          -- opticalFlow  = model.modules[2].modules[2].modules[2].modules[7].output
          -- local imflow = flow2colour(opticalFlow)
          -- table.insert(flowTable, imflow)
      end

      -- local target  = torch.Tensor(opt.batchSize, opt.outputSeqLen, opt.inputSeqLen, opt.imageHeight, opt.imageWidth)
      -- local target = {}
      local input = data[{{}, {opt.inputSeqLen}, {}, {}, {}}]:select(2, 1) 
      -- assert(model.modules[1]:getParameters():sum() == branch_memory.modules[1]:getParameters():sum())
      parameters, grads = model:getParameters() 

      table.insert(inputTable, input)
      local feval = function()
          model:zeroGradParameters()
          local f = 0       
          for i = 1, opt.outputSeqLen do
              local targetTensor =  data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}]:select(2, 1) 
              -- if want to remove this one, set gradloss to be identity function in model
              table.insert(targetTable, targetTensor)
              local output = model:updateOutput(input)
              local outputTensor = output:clone()
              table.insert(outputTable, outputTensor)        

              f = f + criterion:updateOutput(outputTensor, targetTensor)
              -- estimate gradient
              local gradtarget = gradloss:updateOutput(targetTensor):clone()
              local gradoutput = gradloss:updateOutput(outputTensor)
              -- gradients
              local gradErrOutput = criterion:updateGradInput(gradoutput, gradtarget)
              local gradErrGrad = gradloss:updateGradInput(output, gradErrOutput)
              -- local gradOutput = criterion:updateGradInput(output, targetTensor)
              model:updateGradInput(input, gradErrGrad)
              model:accGradParameters(input, gradErrGrad)  

              input = output
              log.trace(string.format('[train] iter %d, frames id %d, loss %.4f', iter, i, loss))
              grads:clamp(-opt.gradClip, opt.gradClip)
              opticalFlow  = clamp.output
              local imflow = flow2colour(opticalFlow)
              table.insert(flowTable, imflow)
          end
          return f, grads
      end

      -- model:updateParameters(opt.lr)
      if math.fmod(iter, 20000) == 0 then
          epoch = epoch + 1
          eta = opt.lr * math.pow(0.5, epoch/50)  
          rmspropconf.learningRate = eta  
      end  

      _,fs = optim.rmsprop(feval, parameters, rmspropconf)
      loss = loss + fs[1]/opt.outputSeqLen

      if(math.fmod(iter, opt.lossDisplayIter) == 0) then
          log.info(string.format('@loss  %.4f iter %d / %d lr %.5f', loss / ( iter - opt.contIter + 1 ), iter, opt.maxIter, eta))
      end


      if(math.fmod(iter, opt.saveIter) == 1 ) then
          log.trace('[saveimg] output')
          OutputToImage(inputTable, iter, 'input')
          OutputToImage(outputTable, iter, 'output')
          -- OutputToImage(outputTable, iter, 'output')
          OutputToImage(targetTable, iter, 'target')
          FlowTableToImage(flowTable, iter, 'flow')
          -- TensorToImage(imflow, iter, 'flow')
      end

      if(math.fmod(iter, 2000) == 1) then
          SaveModel(model, 'model', iter)
          SaveConf(rmspropconf, 'rmspropconf', iter)
      end
      --[[
      err = err + loss

      if math.fmod(iter , opt.nSeq) == 1 then
          log.info('==> iter '.. iter ..', ave loss = ' .. err / (iter) .. ' lr '..opt.lr)
          err = 0
      end
      ]]--
  end
  log.info('@Training done')
  collectgarbage()
end

main()
