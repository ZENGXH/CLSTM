unpack = unpack or table.unpack

require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'HadamardMul'
require 'utils'
require 'cunn'
require 'cutorch'
-- require 'BilinearSamplerBHWD'
-- require 'display_flow'
-- torch.setdefaulttensortype('torch.FloatTensor')
-- require 'DenseTransformer2D'
-- require 'SmoothHuberPenalty'
-- require 'encoder'
-- require 'decoder'
-- require 'flow'

local log = loadfile('log.lua')()
paths.dofile('opts_hko.lua')
dofile('model_hko.lua') 
SaveModel(enc, 'enc', iter)
SaveModel(dec, 'dec', iter)
 
local criterion = nn.SequencerCriterion(nn.MSECriterion())
if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end

local err = 0

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
 
  for iter = 1, opt.maxIter do
      -- define eval closure

      enc:zeroGradParameters()
      dec:zeroGradParameters()
      local sample = datasetSeq[iter]
      local data = sample[1]  
      if opt.useGpu then
          data = data:cuda()
      end

      local inputEncTable = {}
      for i = 1, opt.inputSeqLen do 
          -- log.trace('[append] input data for encoder', i)

          table.insert(inputEncTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
      end

      -- local target  = torch.Tensor(opt.batchSize, opt.outputSeqLen, opt.inputSeqLen, opt.imageHeight, opt.imageWidth)
      local target = {}
      for i = 1, opt.outputSeqLen do
        table.insert(target, data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}] )
      end
      -- local target = data[{{}, {opt.inputSeqLen + 1, opt.totalSeqLen}, {}, {}, {}}] 

      -- in shape (15, 8, 1, 100, 100)
      local output_enc = enc:forward(inputEncTable)

      log.trace('[fw] forward encoder done@ output')
      forwardConnect(enc, dec)
      inputDecTensor = torch.Tensor(opt.batchSize, opt.imageDepth, opt.imageHeight, opt.imageWidth):fill(0)
      if opt.useGpu then
          inputDecTensor = inputDecTensor:cuda()
      end

      local inputDecTable = {}
      for i = 1, opt.outputSeqLen do
          -- log.trace('[append] input data for decoder', i)
          table.insert(inputDecTable, inputDecTensor)
      end

      local output = dec:forward(inputDecTable)

      local loss = criterion:updateOutput(output, target)
      if(math.fmod(iter, 50) == 0) then
          log.info('@loss ', loss, ' iter ', iter, ' / ', opt.maxIter)
      end

      zeroGradDec = {}
      zeroT = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], opt.imageHeight, opt.imageWidth):zero()
      if opt.useGpu then
          zeroT = zeroT:cuda()
      end

      -- log.trace('size of output encoder: ', output_enc[1]:size(),
      --           'size of grad output encoder: ', zeroT:size())
      
      for i = 1, #inputEncTable do
          -- log.trace('[bp] appending tensor to grad output of encoder: ')
          table.insert(zeroGradDec, zeroT)
      end


      gradOutput = criterion:updateGradInput(output, target)
      local gradDec = dec:backward(inputDecTable, gradOutput)
      backwardConnect(enc, dec)
      -- log.trace("gradOutput: ", gradOutput)
      -- log.trace("gradDec", gradDec)

      log.trace("[bp] update grad input of encoder: ")
      local gradEnc = enc:backward(inputEncTable, zeroGradDec)
      
      log.trace("[bp] update parameters")
      dec:updateParameters(opt.lr)
      enc:updateParameters(opt.lr)
      if(math.fmod(iter, 50) == 0) then
          OutputToImage(inputEncTable, iter, 'input')
          log.trace('[saveimg] dec')
          ViewCell(dec, iter, 'dec')
          log.trace('[saveimg] enc')
          ViewCell(enc, iter, 'enc')
          -- ViewCell(output)
          log.trace('[saveimg] output_enc')
          OutputToImage(output_enc, iter, 'output_enc')
          log.trace('[saveimg] output')
          OutputToImage(output, iter, 'output')
      end

      if(math.fmod(iter, 1000) == 0) then
          SaveModel(enc, 'enc', iter)
          SaveModel(dec, 'dec', iter)
      end

      err = err + loss

      if math.fmod(iter , opt.nSeq) == 1 then
          log.info('==> iter '.. iter ..', ave loss = ' .. err / (opt.nSeq) .. ' lr '..opt.lr)
          err = 0
      end
  end
  log.info('@Training done')
  collectgarbage()
end

main()
