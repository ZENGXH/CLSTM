require 'image'
local log = loadfile('log.lua')()
log.level = opt.dataLoaderLogLevel or "trace"
opt.selectStep = opt.selectStep or 1
log.info('[SelectSeq] selectStep set as ', opt.selectStep)


function getdataSeqHko(mode)
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- data size (totalInstances or nsamples=2000?, nSequence_length=20, 1, 64, 64)
   local mode = mode or 'train'
   local datasetSeq ={}
   --------------- configuration: -----------------

    local f 
    if(mode == 'train') then
        f = io.open(opt.listFileTrain, 'r')
        log.trace('opening listFile: ', opt.listFileiTrain)
    elseif(mode == 'test') then
        f = io.open(opt.listFileTest, 'r')
        log.trace('opening listFile: ', opt.listFileiTest)
    else
        log.fatal('[getdataSeqHko] mode: '.. mode..' invalid')
    end

    if(opt.dataOverlap) then
        log.info('[SelectSeq] data overlapping selected, selectStep = ', opt.selectStep)
    else
        log.info('[SelectSeq] data NOT overlapping selected, selectStep = ', opt.selectStep)
    end

    local fileList = {}    
    for line in f:lines() do
        table.insert(fileList, line)
    end

    log.info("[init] read data into fileList table, size in total: ", table.getn(fileList))
    log.info("[init] size of input batch ", opt.batchSize, opt.totalSeqLen, opt.imageDepth, opt.imageHeight, opt.imageWidth)

    function datasetSeq:size()
        return table.getn(fileList)
    end


    local batchStart = 0
    local ind = 1
    function datasetSeq:SelectSeq()
        log.trace("SelectSeq")

        local inputBatch = torch.Tensor(opt.batchSize, opt.totalSeqLen, opt.imageDepth, opt.imageHeight, opt.imageWidth)
        for batch_ind = 1, opt.batchSize do -- filling one batch one by one
            -- math.ceil(torch.uniform(1e-12, nsamples)) 
            -- choose an index in range{1,.. nsamples}
            -- image index
            -- read the 20 frames starting from i

            -- filling one batch
            for frames_id = 1, opt.totalSeqLen do
               -- local imageName = opt.dataPath..fileList[ (batch_ind - 1 + ind) * opt.totalSeqLen + frames_id]
               local imageName = opt.dataPath..fileList[ batchStart + ((frames_id - 1) * opt.selectStep + 1) ]
               inputBatch[batch_ind][frames_id] = image.load(imageName)
               if frames_id == 1 then
                   log.trace("[SelectSeq] load batch#", batch_ind, " seq#",  opt.totalSeqLen, " from ", imageName)
               end
            end

            ind = ind + 1
            if opt.dataOverlap then
                batchStart = (batch_ind - 1) + (ind - 1) * opt.batchSize  
            else -- not overlap for different batch,
                batchStart = (batch_ind - 1 + ind - 1) * opt.batchSize
            end

            if batchStart + opt.totalSeqLen >= #fileList then
                ind = 1
                batchStart = 0
                log.info('[SelectSeq] dataset end, restart')
            end
         end
         return inputBatch, ind
      end

   dataSample = torch.Tensor(opt.batchSize, opt.totalSeqLen, opt.imageDepth, opt.imageHeight, opt.imageWidth)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample, i = self:SelectSeq()
                                       dataSample:copy(sample)
                                       return {dataSample, i}
                                    end})
   return datasetSeq
end

