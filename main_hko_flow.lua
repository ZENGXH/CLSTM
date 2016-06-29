unpack = unpack or table.unpack

require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'utils'
require 'cunn'
require 'cutorch'
require 'display_flow'
require 'optim'

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

-- init model
dofile('model_hko_flow.lua') 
if opt.backend == 'cudnn' then
    require 'cudnn'
    log.info('[init] use cudnn backend')
    model = cudnn.convert(model, cudnn)
    cudnn.fastest = true 
end
-- >>>>>>
-- init LSTM parameters to small values, uniformly distributed
local lstm_params, lstm_grads = model.modules[2].modules[2].modules[1]:getParameters()
lstm_params:uniform(-0.08, 0.08)
-- init LSTM biases to (forget_bias, other_bias)
model.modules[2].modules[2].modules[1]:initBias(0,0)
-- call LSTM forget to reset the memory
model.modules[2].modules[2].modules[1]:forget()
-- <<<<<<<

-- init criterion
local criterion = nn.MSECriterion()
if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end

local err = 0

-- start train
local function main()
  -- cutorch.setDevice(5)
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
  rmspropconf = {learningRate = eta}
  local loss = 0
  for iter = 1, opt.maxIter do
      -- define eval closure

      model:zeroGradParameters()
      model.modules[2].modules[2].modules[1]:forget() -- ConvLSTM forget
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
          log.trace('[branch_memory] input data for encoder', i)
          local framesMemory = data[{{}, {i}, {}, {}, {}}]:select(2, 1)
          branch_memory:updateOutput(framesMemory)
          table.insert(inputTable, framesMemory)

          opticalFlow  = model.modules[2].modules[2].modules[2].modules[7].output
          local imflow = flow2colour(opticalFlow)
          table.insert(flowTable, imflow)
      end

      -- local target  = torch.Tensor(opt.batchSize, opt.outputSeqLen, opt.inputSeqLen, opt.imageHeight, opt.imageWidth)
      -- local target = {}
      local input = data[{{}, {opt.inputSeqLen}, {}, {}, {}}]:select(2, 1) 
     parameters, grads = model:getParameters()

      table.insert(inputTable, input)
      local feval = function()
          model:zeroGradParameters()
          local f = 0       
          for i = 1, opt.outputSeqLen do

              local targetTensor =  data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}]:select(2, 1) 
              table.insert(targetTable, targetTensor)

              local output = model:updateOutput(input)
              f = f + criterion:updateOutput(output, targetTensor)
              local gradOutput = criterion:updateGradInput(output, targetTensor)
              model:updateGradInput(input, gradOutput)
              model:accGradParameters(input, gradOutput)  

              input = output
              local outputTensor = output:clone()
              table.insert(outputTable, outputTensor)
              log.trace(string.format('[train] iter %d, frames id %d, loss %.4f', iter, i, loss))
              grads:clamp(-opt.gradClip,opt.gradClip)

              opticalFlow  = model.modules[2].modules[2].modules[2].modules[7].output
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

      if(math.fmod(iter, 50) == 0) then
          log.info('@loss ', loss/iter, ' iter ', iter, ' / ', opt.maxIter)
      end


      if(math.fmod(iter, opt.saveIter) == 1 ) then
          log.trace('[saveimg] output')
          OutputToImage(inputTable, iter, 'input')
          OutputToImage(outputTable, iter, 'output')
          -- OutputToImage(outputTable, iter, 'output')
          OutputToImage(targetTable, iter, 'target')
          OutputToImage(flowTable, iter, 'flow')
          TensorToImage(imflow, iter, 'flow')
      end

      if(math.fmod(iter, 2000) == 1) then
          SaveModel(model, 'model', iter)
      end

      err = err + loss

      if math.fmod(iter , opt.nSeq) == 1 then
          log.info('==> iter '.. iter ..', ave loss = ' .. err / (iter) .. ' lr '..opt.lr)
          err = 0
      end
  end
  log.info('@Training done')
  collectgarbage()
end

main()