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
dofile(opt.modelFile) 

local backend_name = opt.backend or 'cudnn'
if opt.continueFromPara then
    assert(opt.modelPara)
    assert(opt.contIter)
    model = LoadParametersToModel(model)
else
    opt.conIter = 1
end



-- SaveModel(enc, 'enc', iter)
-- SaveModel(dec, 'dec', iter)

-- init criterion
local criterion = nn.SequencerCriterion(nn.MSECriterion())
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
  datasetSeq = getdataSeqHko('train') -- we sample totalSeqLen consecutive frames
  -- log.info('[init] Loaded ' .. datasetSeq:size() .. ' images')
  log.info('[init] ==> training model')
  
  -- TODO: init parameters of model 

  -- init optimization configure
if opt.useRmsprop then
  rmspropConf = {}
  rmspropConf.weightDecay = opt.weightDecay
  rmspropConf.learningRate = opt.lr 
  -- prepare zero grad for encoder backward
end

  -- prepare zero gradient for encoder backward
  local zeroGradDec = {}
  local zeroT = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], opt.imageHeight, opt.imageWidth):zero()
  if opt.useGpu then
      zeroT = zeroT:cuda()
  end
  for i = 1, opt.inputSeqLen do
      table.insert(zeroGradDec, zeroT)
  end

  -- prepare zero input for decoder forward
  local inputDecTensor = torch.Tensor(opt.batchSize, opt.imageDepth, opt.imageHeight, opt.imageWidth):fill(0)
  if opt.useGpu then
      inputDecTensor = inputDecTensor:cuda()
  end
  local inputDecTable = {}
  for i = 1, opt.outputSeqLen do
      table.insert(inputDecTable, inputDecTensor)
  end

  -- link enc and dec for update parameters
  parameters, grads = model:getParameters()

  -- init loss
  local err = 0
  local loss = 0
  for iter = opt.contIter, opt.maxIter do
      -- define eval closure
      local inputEncTable = {}
      local target = {}
      local output
      local code

      local feval = function() 
          enc:zeroGradParameters()
          dec:zeroGradParameters()
          local sample = datasetSeq[iter]
          local data = sample[1]  
          if opt.useGpu then
              data = data:cuda()
          end

          for i = 1, opt.inputSeqLen do 
              table.insert(inputEncTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
          end
          for i = 1, opt.outputSeqLen do
              table.insert(target, data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}] )
          end
          -- in shape (15, 8, 1, 100, 100)
          code = enc:forward(inputEncTable)
          forwardConnect(enc, dec)
          output = dec:forward(inputDecTable)

          loss = criterion:updateOutput(output, target)
          local gradOutput = criterion:updateGradInput(output, target)

          dec:backward(inputDecTable, gradOutput)
          backwardConnect(enc, dec)
          enc:backward(inputEncTable, zeroGradDec)
          if not opt.useRmsprop then
              dec:updateParameters(opt.lr)
              enc:updateParameters(opt.lr)
          end
        return loss, grads
      end

      if opt.useRmsprop then
          print 'not implement error'
          -- _, fs = optim.rmsprop(feval, parameters, rmspropConf)
          err = err + fs[1] / opt.outputSeqLen
          return 
      else 
          feval()
          -- fs, _ = unpack(feval)
          err = err + loss / opt.outputSeqLen
      end


      if(math.fmod(iter, opt.displayIter) == 0) then
          log.info(string.format('@loss %.4f, iter %d / %d, lr %.6f, param mean %.4f ', err / iter, iter, opt.maxIter, opt.lr, parameters:mean()))
      end

      if(math.fmod(iter, opt.saveIter) == 1) then
          log.trace('[saveimg] output')
          OutputToImage(inputEncTable, iter, 'input')
          OutputToImage(code, iter, 'code')
          OutputToImage(output, iter, 'output')
          if opt.viewCell then
              ViewCell(dec, iter, 'dec')
              ViewCell(enc, iter, 'enc')
          end
      end

      if(math.fmod(iter, opt.modelSaveIter) == 1) then
          SaveModel(model, 'encDec', iter)
      end

  end
  log.info('@Training done')
  collectgarbage()
end

main()
