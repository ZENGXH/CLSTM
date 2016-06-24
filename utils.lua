require 'image'
local log = loadfile('log.lua')()


function OutputToImage(output, iter, name)
    -- output: table, size == outputSeqLength
    assert(type(output) == 'table')
    local output_batch
    for id_frames = 1, #output do
        -- log.trace('saving output image')
        output_batch = output[id_frames]
        if(output_batch:size(1) == opt.batchSize) then
            output_batch = output_batch[1]
        end
        -- log.trace('size: ', output_batch:size())
        if(output_batch:size(1) ~= opt.imageDepth) then 
            -- log.trace(output_batch:size())
            output_batch = output_batch:select(1,1)
            -- log.trace(output_batch:size())
       end

        -- assert(output_batch:size(2) == opt.imageHeight)
        -- assert(output_batch:size(3) == opt.imageWidth)
        -- output_batch: 2, 1, 100, 100
        -- print('size of output_batch: ', output_batch:size())
        image.save(opt.saveDir..timeStamp()..'it'..tostring(iter)..name..'n_'..tostring(id_frames)..'.png', 
                    output_batch:div(output_batch:max()))
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
    local filename = opt.modelDir..modelname..'iter-'..tostring(iter)..'.bin' 
    torch.save(filename, emptymodel)
    log.info('current model is saved as', filename)
end

function PixelToRainfall(img, a, b):
    local a = a or 118.239
    local b = b or 1.5241
    local dBZ = img * 70.0 - 10.0
    local dBR = (dBZ - 10.0 * math.log10(a)):div(b)
    local R = math.pow(10, dBR / 10.0)
    return R
end

function SkillScore(prediction, truth, threthold)
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
    local threshold = threshold pr 0.5
    local bpred = torch.gt(PixelToRainfall(prediction), threshold)
    local btruth = torch.gt(PixelToRainfall(truth), threshold)
    local bpredTrue = bpred:float()
    local btruthTrue = btruth:float()
    local bpredFalse = torch.eq(bpred, 0):float()
    local btruthFalse = torch.eq(btruth, 0):float()
    local hits = bpredTrue:cmul(btruthTrue):sum()
    local misses = bpredFalse:cmul(btruthTrue):sum()
    local falseAlarms = bpredTrue:cmul(btruthFalse):sum()
    local correctNegatives = bpredFalse:cmul(btruthFalse):sum()
    local eps = 1e-9
    local POD = (hits + eps) / (hits + misses + eps)
    local FAR = (falseAlarms) / (hits + falseAlarms + eps)
    local CSI = (hits + eps) / (hits + misses + falseAlarms + eps)
    local correlation = torch.cmul(prediction, truth):sum() / (
                math.sqrt(torch.cmul(prediction, prediction):sum()) * math.sqrt(torch.mul(truth, truth):sum() + eps))
    return {"POD": POD, "FAR": FAR, "CSI": CSI, "correlation": correlation, "Rain RMSE": rain_rmse, "RMSE": rmse}
end



