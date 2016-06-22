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

require 'rnn'
require 'ConvLSTM'
require 'DenseTransformer2D'
require 'SmoothHuberPenalty'
require 'encoder'
require 'decoder'
require 'flow'
require 'stn'

--------- end of build model
local function main()

  -- cutorch.setDevice(1)
  paths.dofile('opts_hko.lua')
  paths.dofile('data_hko.lua')
  paths.dofile('model_hko.lua')
  
  datasetSeq = getdataSeqHko() -- we sample nSeq consecutive frames

  log.info('[init] Loaded ' .. datasetSeq:size() .. ' images')
  log.info('[init] ==> training model')

  torch.manualSeed(opt.seed)
  -- TODO: init parameters of model 
 
  model:training()

  for t = 1,opt.maxIter do
    -- progress
    iter = iter + 1

    -- define eval closure
    local feval = function()
      local f = 0
 
      model:zeroGradParameters()

      inputTable = {}
      
      -- = torch.Tensor(opt.transf,opt.memorySizeH, opt.memorySizeW) 
      sample = datasetSeq[t] -- 
      data = sample[1]  

      local inputSeqLength = opt.inputSeqLength
      local outputSeqLength = opt.outputSeqLength

      for i = 1, input_seqlen do 
        table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):cuda())
      end
      
      target  = torch.Tensor(8, 15, 1, 100, 100)
      target = data[{{}, {input_seqlen+1, 20}, {}, {}, {}}] -- in shape (15, 8, 1, 100, 100)
      -- target:resizeAs(data[1]):copy(data[data:size(1)])
      -- for i = 1, output_seqlen do
      --  table.insert(inputTable, data[i]:cuda())
      -- end    
      target = target:cuda()
      
      -- estimate f and gradients
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
      f = criterion:updateOutput(output, target)
      -- gradients
--      local gradErrOutput = criterion:updateGradInput(gradoutput,gradtarget)
--      local gradErrGrad = gradloss:updateGradInput(output,gradErrOutput)
      
      -- start backward
      local gradInput4 = convForward_4:updateGradInput(inputTable4, f)     
      -- update para:
      convForward_4:accGradParameters(inputTable4, f)  

--- TODO: check BACKWARD CONNECT FOR CONVFORWARD

      local gradInput3 = decoder_3:updateGradInput(output2, gradInput4[2][2])
      -- update para:
      decoder_3:accGradParameters(output2, gradInput4[2][1])
      -- update para:
      decoder_2:accGradParameters(output2, gradInput4[2][2])

      backwardConnect(encoder_0, decoder_2)
      backwardConnect(encoder_1, decoder_3)

-- noinput      local gradInput2 = decoder_2:(output2, gradInput4[2][1])
      -- update para:
-- TODO: ADD BP FOR ENCODER
--      encoder_0:updateGradInput(output2, gradInput4[2][2])





      model:updateGradInput(input4, gradErrGrad)

      model:accGradParameters(inputTable, gradErrGrad)  

      grads:clamp(-opt.gradClip,opt.gradClip)
      return f, grads
    end
   
   
    if math.fmod(t,20000) == 0 then
      epoch = epoch + 1
      eta = opt.eta*math.pow(0.5,epoch/50)    
    end  

    rmspropconf = {learningRate = eta,
                  epsilon = 1e-5,
                  alpha = 0.9}

    _,fs = optim.rmsprop(feval, parameters, rmspropconf)

    err = err + fs[1]
    model:forget()
    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.nSeq) == 1 then
      log.info('==> iter '.. t ..', ave loss = ' .. err / (opt.nSeq) .. ' lr '..eta) -- err/opt.statInterval)
      err = 0
    end
  end
  print ('Training done')
  collectgarbage()
end
main()
