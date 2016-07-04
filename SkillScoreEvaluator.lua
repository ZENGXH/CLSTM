local SkillScoreEvaluator = torch.class 'SkillScoreEvaluator'
local log = loadfile('log.lua')()

function SkillScoreEvaluator:__init(outputSeqLen, outfile, threshold)
    log.outfile = outfile or 'skillScoreEvaluator.log'
    assert(outputSeqLen, 'outputSeqLen for SkillScoreEvaluator required')
    self.scores = {}
    self.outputSeqLen = outputSeqLen
    self.iter = 0
    self.scores.POD = torch.Tensor(outputSeqLen):zero()
    self.scores.FAR = torch.Tensor(outputSeqLen):zero()
    self.scores.CSI = torch.Tensor(outputSeqLen):zero()
    self.scores.correlation = torch.Tensor(outputSeqLen):zero()
    self.scores.rainRmse =  torch.Tensor(outputSeqLen):zero()
    self.scores.rmse = torch.Tensor(outputSeqLen):zero()
    self.threshold = threshold or 0.5
    log.info('[init] SkillScoreEvaluator init done:  ', self.scores, ' threshold: ', self.threshold)
    log.info('SkillScoreEvaluator get outputSeqLen ', self.outputSeqLen)
end


function SkillScoreEvaluator:CalculateSub(prediction, truth, id)
    --[[
    POD = hits / (hits + misses)
    FAR = false_alarms / (hits + false_alarms)
    CSI = hits / (hits + misses + false_alarms)
    ETS = (hits - correct_negatives) / 
            (hits + misses + false_alarms - correct_negatives)
    correlation = 
            (prediction * truth).sum() / 
                (sqrt(square(prediction).sum()) * 
                sqrt(square(truth).sum() + eps)) 
    ]]--
    assert(torch.isTensor(prediction))
    -- log.trace(string.format("prediction: (%.4f, %.4f) vs truth: (%.4f, %.4f)", prediction:max(), prediction:min(), truth:max(), truth:min()))
    -- prediction = prediction:div(prediction:max() - prediction:min())
    local rainfallPred = PixelToRainfall(prediction)
    local rainfallTruth = PixelToRainfall(truth)

    local bpred = torch.gt(rainfallPred, self.threshold)
    local btruth = torch.gt(rainfallTruth, self.threshold)
    
    -- log.trace(string.format('id %d, mean bpred: %.4f, btruth %.4f', id, prediction:mean(), truth:mean())) 
    local bpredTrue = bpred:float()
    local btruthTrue = btruth:float()
    local bpredFalse = torch.eq(bpred, 0):float()
    local btruthFalse = torch.eq(btruth, 0):float()
    local hits = torch.cmul(bpredTrue, btruthTrue):sum()
    local misses = torch.cmul(bpredFalse, btruthTrue):sum()
    
    local falseAlarms = torch.cmul(bpredTrue, btruthFalse):sum()
    local correctNegatives = torch.cmul(bpredFalse, btruthFalse):sum()

    local eps = 1e-9
    local POD = (hits + eps) / (hits + misses + eps)
    local FAR = (falseAlarms) / (hits + falseAlarms + eps)
    local CSI = (hits + eps) / (hits + misses + falseAlarms + eps)
    local correlation = torch.cmul(prediction, truth):sum() / (
                math.sqrt(torch.cmul(prediction, prediction):sum()) * math.sqrt(torch.cmul(truth, truth):sum() + eps))
  
    local rainRmse = nn.MSECriterion():updateOutput(rainfallPred, rainfallTruth)
    
    local rmse =  nn.MSECriterion():updateOutput(prediction, truth)
    -- log.trace(string.format("id: %d POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f", id, POD, FAR, CSI, correlation, rainRmse))
    self.scores.POD[id] = self.scores.POD[id] + POD
    self.scores.FAR[id] = self.scores.FAR[id] + FAR
    self.scores.CSI[id] = self.scores.CSI[id] + CSI
    self.scores.correlation[id] = self.scores.correlation[id] + correlation
    self.scores.rainRmse[id] = self.scores.rainRmse[id] + rainRmse
    self.scores.rmse[id] = self.scores.rmse[id] + rmse
end

local function PixelToRainfall(img, a, b)
    local a = a or 118.239
    local b = b or 1.5241
    local dBZ = img * 70.0 - 10.0
    local dBR = (dBZ - 10.0 * math.log10(a)):div(b)
    local R = torch.pow(10, dBR / 10.0)
    return R
end

function SkillScoreEvaluator:Update(prediction, truth)
    -- prediction, truth are table
    self.iter = self.iter + 1

    assert(type(prediction) == 'table')
    log.trace(string.format("eval all output in length: %d", opt.outputSeqLen))
    for id = 1, opt.outputSeqLen do 
        self:CalculateSub(prediction[id], truth[id], id)
    end
    return self.scores
end

function SkillScoreEvaluator:ResetThreshold(threshold)
    log.info('[SkillScoreEvaluator] ResetThreshold from ', self.threshold, ' -> ', threshold)
    self.threshold = threshold
end

function SkillScoreEvaluator:PrintAverage()
    local iter = self.iter
    local POD = self.scores.POD[self.outputSeqLen] / iter
    local FAR = self.scores.FAR[self.outputSeqLen] / iter
    local CSI = self.scores.CSI[self.outputSeqLen] / iter
    local correlation = self.scores.correlation[self.outputSeqLen] / iter
    local rainRmse = self.scores.rainRmse[self.outputSeqLen] / iter
    local rmse = self.scores.rmse[self.outputSeqLen] / iter

    log.info(string.format("[score] iter %d @s%d ave POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f \t rmse %.3f", iter, self.outputSeqLen, POD, FAR, CSI, correlation, rainRmse, rmse))
end

function SkillScoreEvaluator:Summary()
    local maxTestIter = self.iter
    local allPOD = self.scores.POD:div(maxTestIter)
    local allFAR = self.scores.FAR:div(maxTestIter)
    local allCSI = self.scores.CSI:div(maxTestIter)
    local allCorrelation = self.scores.correlation:div(maxTestIter)
    local allRainRmse = self.scores.rainRmse:div(maxTestIter)
    local allRmse = self.scores.rmse:div(maxTestIter)

    log.info("[SkillScoreEvaluator] Summary for total iter ", maxTestIter)
    log.info("POD: ", allPOD)
    log.info("FAR: ", allFAR)
    log.info("CSI; ", allCSI)
    log.info("correlation: ", allCorrelation)
    log.info("rainRmse: ", allRainRmse)
    log.info("rmse: ", allRmse)

    log.info("@done evaluatei & save")
    
end

function SkillScoreEvaluator:Save(path)
    log.info('[SkillScoreEvaluator] save at '..path..'eval_score.bin')
    torch.save(path..'eval_score.bin', self.scores)
end

function SkillScoreEvaluator:__tostring__()
    print('SkillScoreEvaluator threshold: ', self.threshold, ' for outputSeqLen ', self.outputSeqLen)
    print('[contain] POD \t FAR \t CSI \t correlation \t rainRmse \t rmse')
end
