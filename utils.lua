require 'image'
local log = loadfile('log.lua')()

function OutputToImage(output, iter, name, saveDir)
    local saveDir = saveDir or opt.saveDir
    -- output: table, size == outputSeqLength
    assert(type(output) == 'table')
    local output_batch
    for id_frames = 1, #output do
        output_batch = output[id_frames]
        if(output_batch:size(1) == opt.batchSize) then -- choose the first batch
            output_batch = output_batch[1]
        end
        if(output_batch:size(1) ~= opt.imageDepth and output_batch:dim() == 3) then -- choose the first dimension 
            output_batch = output_batch:select(1,1)
        end
        if(output_batch:dim() == 4 and output_batch:size(1) == 1) then -- target: {(1, 1, 100, 100). (1, 1, 100, 100), ...}
            output_batch = output_batch[1]
        end
        image.save(saveDir..'it'..tostring(iter)..name..'n_'..tostring(id_frames)..'.png', output_batch:div(output_batch:max()))
    end
end

function WeightInit(net)
    log.trace('[WeightInit]')    
    net:reset(Xavier(net.nInputPlane * net.kH * net.kW, 
                    net.nOutputPlane * net.kH * net.kW))
--[[
    for i = 1, #net do
        log.trace('[WeightInit] id #', i)    
        m:reset(xavier(m.nInputPlane * m.kH * m.kW, m.nOutputPlane * m.kH * m.kW))
    end
]]--
end

function Xavier(inputSize, outputSize)
    return math.sqrt(2/(inputSize + outputSize))
end

function ViewCell(net, iter, name)
    local lstm
    for i = 1, #net.lstmLayers do
        lstm = net.lstmLayers[i]
        local finalStep = #lstm.cells
        local cell = lstm.cells[finalStep] -- {}
        assert(torch.isTensor(cell)) -- size 17 100 100
        _savegroup(cell, name..'cell_'..tostring(finalStep), iter)
        _savegroup(lstm.cells[1], name..'cell_'..tostring(1), iter)
    end
end

function _savegroup(figure, name, iter)
    local img = figure:clone()
    local depth
    if(type(figure) == 'table') then
        depth = #figure
        log.trace('<saveGroup> get a table in size ', #figure)
    elseif(torch.isTensor(figure)) then
        depth = figure:size(1)
        img = img:double()
        --log.trace('<saveGroup> ', img:size())
        if(img:dim() == 4) then
            img = img[1]
        end
    else
        depth = 1
    end
    
    local imgName = opt.saveDir..timeStamp()..'-it-'..tostring(iter)..'-'..name..'.png'
    local img_group = image.toDisplayTensor{input = img,
                                            padding = 2,
                                            nrow = math.floor(math.sqrt(depth)),
                                            symmetric = true}
    image.save(imgName, img_group / img_group:max())
end

function timeStamp()
      return os.date("%H%M")
      -- return os.date("%m%d%H%M%S")
end

function SaveModel(model, modelname, iter)
    local emptymodel = model:clone('weight', 'bias'):float()
    local filename = opt.modelDir..modelname..'_iter_'..tostring(iter)..'.bin' 
    torch.save(filename, emptymodel)
    log.info('current model is saved as', filename)
end

function PixelToRainfall(img, a, b)
    local a = a or 118.239
    local b = b or 1.5241
    local dBZ = img * 70.0 - 10.0
    local dBR = (dBZ - 10.0 * math.log10(a)):div(b)
    local R = torch.pow(10, dBR / 10.0)
    return R
end


function SkillScoreSub(scores, prediction, truth, threshold, id)
    
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
    assert(threshold)
    assert(torch.isTensor(prediction))
    -- log.trace(string.format("prediction: (%.4f, %.4f) vs truth: (%.4f, %.4f)", prediction:max(), prediction:min(), truth:max(), truth:min()))
    -- prediction = prediction:div(prediction:max() - prediction:min())
    local rainfallPred = PixelToRainfall(prediction)
    local rainfallTruth = PixelToRainfall(truth)

    local bpred = torch.gt(rainfallPred, threshold)
    local btruth = torch.gt(rainfallTruth, threshold)
    
    -- log.trace(string.format('id %d, mean bpred: %.4f, btruth %.4f', id, prediction:mean(), truth:mean())) 
    local bpredTrue = bpred:float()
    local btruthTrue = btruth:float()
    local bpredFalse = torch.eq(bpred, 0):float()
    local btruthFalse = torch.eq(btruth, 0):float()
    local hits = torch.cmul(bpredTrue, btruthTrue):sum()
    local misses = torch.cmul(bpredFalse, btruthTrue):sum()
    
    local falseAlarms = torch.cmul(bpredTrue, btruthFalse):sum()
    log.trace('falseAlarms: bpredTrue ', bpredTrue:mean())
    log.trace('falseAlarms: btruthFalse ', btruthFalse:mean())
    log.trace('falseAlarms: ', falseAlarms)
    local correctNegatives = torch.cmul(bpredFalse, btruthFalse):sum()

    local eps = 1e-9
    local POD = (hits + eps) / (hits + misses + eps)
    local FAR = (falseAlarms) / (hits + falseAlarms + eps)
    local CSI = (hits + eps) / (hits + misses + falseAlarms + eps)
    local correlation = torch.cmul(prediction, truth):sum() / (
                math.sqrt(torch.cmul(prediction, prediction):sum()) * math.sqrt(torch.cmul(truth, truth):sum() + eps))
    local rainRmse = torch.cmul((rainfallPred - rainfallTruth), (rainfallPred - rainfallTruth)):sum()
    rainRmse = rainRmse / prediction:nElement()
    
    -- log.trace(string.format("id: %d POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f", id, POD, FAR, CSI, correlation, rainRmse))
    scores.POD[id] = scores.POD[id] + POD
    scores.FAR[id] = scores.FAR[id] + FAR
    scores.CSI[id] = scores.CSI[id] + CSI
    scores.correlation[id] = scores.correlation[id] + correlation
    scores.rainRmse[id] = scores.rainRmse[id] + rainRmse
    return scores
end

function SkillScore(scores, prediction, truth, threshold)
    local threshold = threshold or 0.5
    assert(type(prediction) == 'table')
    log.trace(string.format("eval all output in length: %d", opt.outputSeqLen))
    for id = 1, opt.outputSeqLen do 
        scores = SkillScoreSub(scores, prediction[id], truth[id], threshold, id)
    end
    return scores
end


function forwardConnect(enc, dec)
    -- log.trace('[forwardConnect]')
    for i=1, #enc.lstmLayers do
        if opt.useSeqLSTM then
            dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].output[opt.inputSeqLen]
            dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cell[opt.inputSeqLen]
        else
            -- log.info("dec connect: ", #enc.lstmLayers[i].cells)
            assert(#enc.lstmLayers[i].cells == opt.inputSeqLen)
            dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[opt.inputSeqLen])
            dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[opt.inputSeqLen])
        end
    end
    --log.trace('[forwardConnect] done@')
end

function backwardConnect(enc, dec)
    -- log.trace('[backwardConnect]')
    for i=1,#enc.lstmLayers do
        if opt.useSeqLSTM then
            enc.lstmLayers[i].userNextGradCell = dec.lstmLayers[i].userGradPrevCell
            enc.lstmLayers[i].gradPrevOutput = dec.lstmLayers[i].userGradPrevOutput
        else
            enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
            enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
        end
    end
end
