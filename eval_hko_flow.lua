require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'utils'
require 'cunn'
require 'cutorch'

local evalLog = loadfile('log.lua')()
paths.dofile('opts_hko.lua')
evalLog.level = opt.evalLogLevel or "info"
paths.dofile('data_hko.lua')
evalLog.outfile = opt.testLogDir..'eval_hko_log'

startTestUtils()
assert(opt.init)
opt.useGpu = false 
evalLog.info('[init] useGpu: ', opt.useGpu)

-- set criterion
local criterion = nn.MSECriterion()
if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end
evalLog.info('[init] set criterion ', criterion)

local err = 0

-- load model
local model = torch.load(opt.modelFlow)
local lstm = model.modules[2].modules[2].modules[1]
local enc_conv = model.modules[1]
local branch_memory = nn.Sequential():add(enc_conv):add(lstm)
model = model:double()
if opt.useGpu then
    model = model:cuda()
end
evalLog.info(string.format("[init] ==> evaluating hko flow model: %s ", opt.modelFlow))

-- clear grad 
model:zeroGradParameters()

-- prepare datasetSeq
datasetSeqTest = getdataSeqHko('test')

local scores = {}
scores.POD = torch.Tensor(opt.outputSeqLen):zero()
scores.FAR = torch.Tensor(opt.outputSeqLen):zero()
scores.CSI = torch.Tensor(opt.outputSeqLen):zero()
scores.correlation = torch.Tensor(opt.outputSeqLen):zero()
scores.rainRmse =  torch.Tensor(opt.outputSeqLen):zero()
local POD = 0
local FAR = 0
local CSI = 0
local correlation = 0
local rainRmse = 0

for iter = 1, opt.maxTestIter do
    -- init table var
    local inputTable = {}
    local inputEncTable = {}
    local opticalFlow
    local imflow
    local flowTable = {}
    local outputTable = {}
    local targetTable = {}

    -- clear memory, load current data
    lstm:forget()
    local sample = datasetSeqTest[iter]
    local data = sample[1]  
    if opt.useGpu then
        data = data:cuda()
    end

    -- first memorize the first opt.inputSeqLen - 1 frames
      for i = 1, opt.inputSeqLen - 1 do 
          log.trace('[branch_memory] input data for encoder', i)
          local framesMemory = data[{{}, {i}, {}, {}, {}}]:select(2, 1)
          branch_memory:updateOutput(framesMemory)
          table.insert(inputTable, framesMemory)
          opticalFlow  = model.modules[2].modules[2].modules[2].modules[7].output
          local imflow = flow2colour(opticalFlow)
          table.insert(flowTable, imflow)
      end
      
    -- start prediction

    local input = data[{{}, {opt.inputSeqLen}, {}, {}, {}}]:select(2, 1) 
    local f = 0       
    for i = 1, opt.outputSeqLen do

        local targetTensor =  data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}]:select(2, 1) 
        table.insert(targetTable, targetTensor)
        local output = model:updateOutput(input)
        f = f + criterion:updateOutput(output, targetTensor)
        input = output
        local outputTensor = output:clone()
        table.insert(outputTable, outputTensor)
        log.trace(string.format('[train] iter %d, frames id %d, loss %.4f', iter, i, loss))

        opticalFlow  = model.modules[2].modules[2].modules[2].modules[7].output
        local imflow = flow2colour(opticalFlow)
        table.insert(flowTable, imflow)
    end



    -- calculate scores 
    scores = SkillScore(scores, outputTable, targetTable, opt.threthold)
    POD = scores.POD[opt.outputSeqLen] / iter
    FAR = scores.FAR[opt.outputSeqLen] / iter
    CSI = scores.CSI[opt.outputSeqLen] / iter
    correlation = scores.correlation[opt.outputSeqLen] / iter
    rainRmse = scores.rainRmse[opt.outputSeqLen] / iter

    if(math.fmod(iter, opt.testSaveIter) == 1) then
        evalLog.trace('[saveimg] input, output and target save to %s')
        OutputToImage(inputEncTable, iter, 'input')
        OutputToImage(output, iter, 'output')
        OutputToImage(target, iter, 'target')
    end


    evalLog.info(string.format("iter: %d POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f", iter, POD, FAR, CSI, correlation, rainRmse))
end

allPOD = scores.POD:div(opt.maxTestIter)
allFAR = scores.FAR:div(opt.maxTestIter)
allCSI = scores.CSI:div(opt.maxTestIter)
allCorrelation = scores.correlation:div(opt.maxTestIter)
allRainRmse = scores.rainRmse:div(opt.maxTestIter)


evalLog.info("POD: ", allPOD)
evalLog.info("FAR: ", allFAR)
evalLog.info("CSI; ", allCSI)
evalLog.info("correlation: ", allCorrelation)
evalLog.info("rainRmse: ", allCorrelation)
evalLog.info("@done evaluate")
