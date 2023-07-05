classdef Transformer < matlab.mixin.Copyable
    properties
        embedding
        encoderLayers
    end
    
    methods
        function obj = Transformer(inputDim, hiddenDim, numLayers, numHeads)
            obj.embedding = embeddingLayer(hiddenDim, inputDim);
            obj.encoderLayers = repmat(EncoderLayer(hiddenDim, numHeads), 1, numLayers);
        end
        
        function encoded = forward(obj, x)
            embedded = obj.embedding(x);
            encoded = embedded;
            
            for i = 1:numel(obj.encoderLayers)
                encoded = obj.encoderLayers(i).forward(encoded);
            end
        end
    end
end

classdef EncoderLayer < matlab.mixin.Copyable
    properties
        multiheadAttention
        feedForward
    end
    
    methods
        function obj = EncoderLayer(hiddenDim, numHeads)
            obj.multiheadAttention = MultiheadAttention(hiddenDim, numHeads);
            obj.feedForward = FeedForward(hiddenDim);
        end
        
        function encoded = forward(obj, x)
            attended = obj.multiheadAttention.forward(x, x, x);
            encoded = obj.feedForward.forward(attended);
        end
    end
end

classdef MultiheadAttention < matlab.mixin.Copyable
    properties
        numHeads
        headDim
        qLinear
        kLinear
        vLinear
        outLinear
    end
    
    methods
        function obj = MultiheadAttention(hiddenDim, numHeads)
            obj.numHeads = numHeads;
            obj.headDim = hiddenDim / numHeads;
            
            obj.qLinear = fullyConnectedLayer(hiddenDim);
            obj.kLinear = fullyConnectedLayer(hiddenDim);
            obj.vLinear = fullyConnectedLayer(hiddenDim);
            obj.outLinear = fullyConnectedLayer(hiddenDim);
        end
        
        function attended = forward(obj, query, key, value)
            batchSize = size(query, 1);
            
            q = obj.qLinear.forward(query);
            k = obj.kLinear.forward(key);
            v = obj.vLinear.forward(value);
            
            q = reshape(q, [batchSize, obj.numHeads, obj.headDim]);
            k = reshape(k, [batchSize, obj.numHeads, obj.headDim]);
            v = reshape(v, [batchSize, obj.numHeads, obj.headDim]);
            
            scores = (q * k') / sqrt(obj.headDim);
            attention = softmax(scores, 'dim', -1);
            
            attended = attention * v';
            attended = reshape(attended, [batchSize, obj.headDim * obj.numHeads]);
            attended = obj.outLinear.forward(attended);
        end
    end
end

classdef FeedForward < matlab.mixin.Copyable
    properties
        linear1
        linear2
    end
    
    methods
        function obj = FeedForward(hiddenDim)
            obj.linear1 = fullyConnectedLayer(hiddenDim * 4);
            obj.linear2 = fullyConnectedLayer(hiddenDim);
        end
        
        function x = forward(obj, x)
            x = obj.linear1.forward(x);
            x = relu(x);
            x = obj.linear2.forward(x);
        end
    end
end

% Usage example
inputDim = 1000;
hiddenDim = 256;
numLayers = 6;
numHeads = 8;

model = Transformer(inputDim, hiddenDim, numLayers, numHeads);
inputData = [1, 2, 3, 4, 5; 6, 7, 8, 9, 10];  % Example input data
output = model.forward(inputData);
disp(size(output));
