require 'image'
local log = loadfile('log.lua')()


function OutputToImage(output, iter, name)
    -- output: table, size == outputSeqLength
    assert(type(output) == 'table')
    local output_batch
    for id_frames = 1, #output do
        log.trace('saving output image')
        output_batch = output[1][1]
        assert(output_batch:size(1) == opt.imageDepth)
        assert(output_batch:size(2) == opt.imageHeight)
        assert(output_batch:size(3) == opt.imageWidth)
        -- output_batch: 2, 1, 100, 100
        print('size of output_batch: ', output_batch:size())
        image.save(opt.saveDir..'it'..tostring(iter)..name..'n_'..tostring(id_frames)..'.png', 
                    output_batch)
    end
end

function WeightInit(net)
    log.trace('[WeightInit]')    
    net:reset(xavier(net.nInputPlane * net.kH * net.kW, 
                    net.nOutputPlane * net.kH * net.kW))
--[[
    for i = 1, #net do
        log.trace('[WeightInit] id #', i)    
        m:reset(xavier(m.nInputPlane * m.kH * m.kW, m.nOutputPlane * m.kH * m.kW))
    end
]]--
end

function xavier(inputSize, outputSize)
    return math.sqrt(2/(inputSize + outputSize))
end

function ViewCell(net)
    local lstm
    for i = 1, #net.lstmLayers do
        lstm = net.lstmLayers[1]
        cell
