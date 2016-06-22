unpack = unpack or table.unpack

require 'nn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
require 'image'
require 'stn'
require 'BilinearSamplerBHWD'
require 'optim'
require 'display_flow'

--torch.setdefaulttensortype('torch.FloatTensor')
-------- build model

require 'ConvLSTM'
require 'DenseTransformer2D'
require 'SmoothHuberPenalty'
require 'encoder'
require 'decoder'
require 'flow'

--------- end of build model
local function main()

  -- cutorch.setDevice(1)
  paths.dofile('opts_hko.lua')
  log.trace("load opt ")

  paths.dofile('data_hko.lua')
  model = loadfile('model_hko.lua') 
  datasetSeq = getdataSeqHko() -- we sample nSeq consecutive frames

  log.info('[init] Loaded ' .. datasetSeq:size() .. ' images')
  log.info('[init] ==> training model')

  torch.manualSeed(opt.seed)
  -- TODO: init parameters of model 
 
  model:training()

  for t = 1,opt.maxIter do
      iter = iter + 1
      -- define eval closure
      model:zeroGradParameters()

      inputTable = {}
      sample = datasetSeq[t] -- 
      data = sample[1]  

      local inputSeqLength = opt.inputSeqLength
      local outputSeqLength = opt.outputSeqLength

      for i = 1, inputSeqLen do 
          table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):cuda())
      end

      target  = torch.Tensor(opt.batchSize, opt.outputSeqLen, opt.inputSeqLen, opt.imageHeight, opt.imageWidth)

      target = data[{{}, {opt.inputSeqLen + 1, 20}, {}, {}, {}}] -- in shape (15, 8, 1, 100, 100)
      output = model:updateOutput(inputTable)

      local output0 = encoder_0:updateOutput(inputTable)
      local output1 = encoder_1:updateOutput(output1)
      forwardConnect(encoder_0, decoder_2)
      forwardConnect(encoder_1, decoder_3)
      local inputTable2 = {encoder_0.userPrevOutput, encoder_0.userPrevCell}
      local output2 = decoder_2:updateOutput(inputTable2)
      local output3 = decoder_3:updateOutput(output2)
      local inputTable4 = {{output0(-1), output1(-2)},{output2, output3}}
      local output = convForward_4:updateOutput(input4)


      -- gradtarget = gradloss:updateOutput(target):clone()
      -- gradoutput = gradloss:updateOutput(output)

      -- f = f + criterion:updateOutput(gradoutput,gradtarget)
      loss = criterion:updateOutput(output, target)

      gradOutput = criterion:updateGradInput()

      -- start backward
      local gradInput4 = convForward_4:updateGradInput(inputTable4, loss)     
      -- update para:
      convForward_4:accGradParameters(inputTable4, loss)  
      -- TODO: check BACKWARD CONNECT FOR CONVFORWARD

      local gradInput3 = decoder_3:updateGradInput(output2, gradInput4[2][2])

      -- update para:
      decoder_3:accGradParameters(output2, gradInput4[2][1])
      decoder_2:accGradParameters(output2, gradInput4[2][2])

      backwardConnect(encoder_0, decoder_2)
      backwardConnect(encoder_1, decoder_3)
   
      model:updateGradInput(input4, gradErrGrad)
      model:accGradParameters(inputTable, gradErrGrad)  

      err = err + loss
      model:forget()

      if math.fmod(t , opt.nSeq) == 1 then
          log.info('==> iter '.. t ..', ave loss = ' .. err / (opt.nSeq) .. ' lr '..eta)
          err = 0
      end
  end
  log.info('@Training done')
  collectgarbage()
end
