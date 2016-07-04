require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'utils'
require 'cunn'
require 'cutorch'
require 'SkillScoreEvaluator' 
local log = loadfile('log.lua')()
paths.dofile('opts_hko.lua')
log.level = opt.evalLogLevel or "info"
paths.dofile('data_hko.lua')
log.outfile = opt.testLogDir..'eval_hko_log'
if not paths.dirp(opt.testLogDir) then
    os.execute('mkdir -p ' ..opt.testLogDir )
end
log.info('[init] load model file: ', opt.modelFile)
log.info('[init] load model parameters: ', opt.modelPara)

dofile(opt.modelFile)
parameters, gradParameters = model:getParameters()
log.info('[init] para of model init: ', parameters:sum())

model = LoadParametersToModel(model)

parameters, gradParameters = model:getParameters()
log.info('[init] para of model after set: ', parameters:sum())

local paraEnc, _ = enc:getParameters()
local paraDec, _ = dec:getParameters()
log.info('sum of para: ', paraEnc:sum(), paraDec:sum(), parameters:sum())

startTestUtils()
assert(opt.init)
-- opt.useGpu = true 
log.info('[init] useGpu: ', opt.useGpu)

local criterion = nn.SequencerCriterion(nn.MSECriterion())
log.info('[init] set criterion ', criterion)

if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end
-- local model = torch.load(opt.modelEncDec)

-- enc = enc:double()
-- dec = dec:double()

enc:zeroGradParameters()
dec:zeroGradParameters()


log.info(string.format("[init] ==> evaluating model:  %s", opt.modelPara))

datasetSeq = getdataSeqHko('test')
SkillScore = SkillScoreEvaluator(opt.outputSeqLen)

-- prepare zero input for decoder:
--
  local inputDecTensor = torch.Tensor(opt.batchSize, opt.imageDepth, opt.inputSizeH, opt.inputSizeW):fill(0)
if opt.useGpu then
    inputDecTensor = inputDecTensor:cuda()
end
local inputDecTable = {}
for i = 1, opt.outputSeqLen do
    table.insert(inputDecTable, inputDecTensor)
end
local loss = 0
for iter = 1, opt.maxTestIter do
    local sample = datasetSeq[iter]
    local data = sample[1]  
    if opt.useGpu then
        data = data:cuda()
    end

    local inputEncTable = {}
    for i = 1, opt.inputSeqLen do 
        table.insert(inputEncTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
    end
    local target = {}
    for i = 1, opt.outputSeqLen do
        table.insert(target, data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}] )
    end

    -- in shape (15, 8, 1, 100, 100)
    local output_enc = enc:forward(inputEncTable)
    forwardConnect(enc, dec)
    
    local output = dec:forward(inputDecTable)
    SkillScore:Update(output, target)
    if(math.fmod(iter, opt.testSaveIter) == 1) then
        log.trace('[saveimg] input, output and target save to ', opt.imgDir)
        OutputToImage(inputEncTable, iter, 'input')
        OutputToImage(output, iter, 'output')
        OutputToImage(target, iter, 'target')
    end
    
    loss = loss + criterion:updateOutput(output, target) / opt.outputSeqLen

    if(math.fmod(iter, opt.scoreDisplayIter) == 0) then
        SkillScore:PrintAverage()
    end
end
SkillScore:Summary()
SkillScore:Save(opt.modelDir)
