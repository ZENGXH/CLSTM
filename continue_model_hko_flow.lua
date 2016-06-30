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
require 'flow'
require 'BilinearSamplerBHWD'
require 'DenseTransformer2D' -- AffineGridGeneratorOpticalFlow2D


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
if opt.useGpu and opt.backend == 'cudnn' then
    require 'cudnn'
end

-- init model
dofile('model_hko_flow.lua')

if not opt.continueFromPara then
    -- dofile('model_hko_flow.lua') 
    log.info(string.format("[init] ==> evaluating hko flow model: %s ", opt.modelFlow))
    local modelLoad = torch.load(opt.modelFlow)
    log.info('[init] model: ', model)
    local p, g = model:parameters()
    local p2, g2 = modelLoad:parameters()
    for i = 1, #p do
        p[i]:fill(1):cmul(p2[i]:cuda())
    end
    modelLoad = {}
    p2 = {}
    g2 = {}
else
    -- #TODO: load parameters
end

parameters, grads = model:getParameters()
log.info('size not match: '..tostring(grads:size(1)).." != "..tostring(parameters:size(1)))
assert(grads:size(1) == parameters:size(1), 'size not match: '..tostring(grads:size(1)).." != "..tostring(parameters:size(1)))

local lstm = model.modules[2].modules[2].modules[1]
log.info('[test] model: ', model)
log.info('[test] lstm ', lstm)
local enc_conv = model.modules[1]
local branch_memory = nn.Sequential():add(enc_conv):add(lstm)
-- model = model:double()
if opt.useGpu then
    model = model:cuda()
end
log.info('conv', enc_conv)
log.info('branch_memory: ', branch_memory)

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
  for iter = opt.contIter, opt.maxIter do
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
      end

      -- local target  = torch.Tensor(opt.batchSize, opt.outputSeqLen, opt.inputSeqLen, opt.imageHeight, opt.imageWidth)
      -- local target = {}
      local input = data[{{}, {opt.inputSeqLen}, {}, {}, {}}]:select(2, 1) 
      -- log.info('size not match: '..tostring(grads:size(1)).." != "..tostring(parameters:size(1)))
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

      if(math.fmod(iter, opt.lossDisplayIter) == 1) then
          log.info('@loss ', loss / (iter - opt.contIter + 1) , ' iter ', iter, ' / ', opt.maxIter)
      end


      if(math.fmod(iter, opt.saveIter) == 1 ) then
          log.trace('[saveimg] output')
          OutputToImage(inputTable, iter, 'input')
          OutputToImage(outputTable, iter, 'output')
          -- OutputToImage(outputTable, iter, 'output')
          OutputToImage(targetTable, iter, 'target')
          OutputToImage(flowTable, iter, 'flow')
          -- OutputToImage(imflow, iter, 'flow')
      end

      if(math.fmod(iter, 2000) == 1) then
          SaveModel(model, 'model', iter)
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
