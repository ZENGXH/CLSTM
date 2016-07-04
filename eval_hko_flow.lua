require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'utils'
require 'ConvLSTM'
require 'utils'
require 'flow'
require 'BilinearSamplerBHWD'
require 'DenseTransformer2D' -- AffineGridGeneratorOpticalFlow2D
require 'display_flow'
require 'SkillScoreEvaluator'
paths.dofile('opts_hko.lua')
if opt.useGpu then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(4)
end
local evalLog = loadfile('log.lua')()
evalLog.level = opt.evalLogLevel or "info"
paths.dofile('data_hko.lua')
evalLog.outfile = opt.testLogDir..'eval_hko_log'

startTestUtils()
assert(opt.init)
-- opt.useGpu = false 
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
-- evalLog.info(string.format("[init] ==> evaluating hko flow model: %s ", opt.modelFlow))
if opt.useGpu and opt.backend == 'cudnn' then
    require 'cudnn'
end
-- local model = torch.load(opt.modelFlow)
evalLog.info('[init] load model file: ', opt.modelFile)
evalLog.info('[init] load model parameters: ', opt.modelPara)
dofile(opt.modelFile)
model = LoadParametersToModel(model)
parameters, gradParameters = model:getParameters()
evalLog.trace('[init] para of model init: ', parameters:sum())
--[[
local para = torch.load(opt.modelPara):cuda()
evalLog.trace('[init] para of model load: ', para:sum())
parameters:fill(1):cmul(para)

-- perform check:
parameters, gradParameters = model:getParameters()
evalLog.trace('[init] para of model after set: ', parameters:sum())
evalLog.info('[test] model: ', model)
]]--

-- get module from model
-- local enc_conv = model.modules[1]

if opt.useGpu then
    model = model:cuda()
end

evalLog.info('[test] lstm ', lstm)
-- clear grad 
model:zeroGradParameters()

-- prepare datasetSeq
datasetSeqTest = getdataSeqHko('test')
SkillScore = SkillScoreEvaluator(opt.outputSeqLen)
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
          evalLog.trace('[branch_memory] input data for encoder', i)
          local framesMemory = data[{{}, {i}, {}, {}, {}}]:select(2, 1)
          model:updateOutput(framesMemory)
          table.insert(inputTable, framesMemory)
          opticalFlow  = clamp.output
          local imflow = flow2colour(opticalFlow)
          table.insert(flowTable, imflow)
      end
      
    -- start prediction

    local input = data[{{}, {opt.inputSeqLen}, {}, {}, {}}]:select(2, 1) 

    table.insert(inputTable, input)
    local f = 0       
    for i = 1, opt.outputSeqLen do
        local targetTensor =  data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}]:select(2, 1) 
        table.insert(targetTable, targetTensor)
        local output = model:updateOutput(input)
        f = f + criterion:updateOutput(output, targetTensor)
        input = output
        local outputTensor = output:clone()
        table.insert(outputTable, outputTensor)
        evalLog.trace(string.format('[train] iter %d, frames id %d, loss %.4f', iter, i, f))
        opticalFlow  = clamp.output
        local imflow = flow2colour(opticalFlow)
        table.insert(flowTable, imflow)
    end
    -- calculate scores 
    SkillScore:Update(outputTable, targetTable)

    if(math.fmod(iter, opt.testSaveIter) == 1) then
        evalLog.trace('[saveimg] input, output and target save to %s')
        OutputToImage(inputEncTable, iter, 'input')
        OutputToImage(outputTable, iter, 'output')
        OutputToImage(targetTable, iter, 'target')
        FlowTableToImage(flowTable, iter, 'flow')
    end
    SkillScore:PrintAverage(iter)
    evalLog.info(string.format("iter: %d  loss %.4f", iter, f / (opt.outputSeqLen * iter) ))
end

SkillScore:Summary(iter)
SkillScore:Save(opt.testLogDir)
