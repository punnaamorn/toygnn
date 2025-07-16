begin
	using Pkg
	using Graphs
	using MolecularGraph
	using GraphNeuralNetworks
	using RDKitMinimalLib
	using Flux
	using CSV
	using DataFrames

	#create the node data in both datastore and vector format, and indicator vector
	function verPropTest(allSmiles)
		atomNum, charge, lonePair = [], [], []
		indicator, i = [], 1
		for smiles in allSmiles
			mol = smilestomol(smiles)

			#indicator
			atomCount = size(atom_number(mol), 1)
			append!(indicator, fill(i, atomCount))
			i += 1

			#nodes
			append!(atomNum, atom_number(mol))
			append!(charge, atom_charge(mol))
			append!(lonePair, lone_pair(mol))

			#edge and graph
		end
		vector = [atomNum',
				  charge',
				  lonePair']
		vector = reduce(vcat, vector)
		
		ndata = DataStore(atomicNumber = (atomNum), charge = (charge), 
					   lonePair = (lonePair))
		
		return ndata, vector, indicator
	end

	#put adjacency matrices together diagonally
	function adjMat(matrices)
		totalRows = sum(size(mat, 1) for mat in matrices)
		totalCols = sum(size(mat, 2) for mat in matrices)
		result = zeros(Int, totalRows, totalCols)
		iRow = 0
		iCol = 0
		for currMat in matrices
			r, c = size(currMat)
			result[iRow + 1: iRow + r, iCol + 1: iCol + c] .= currMat
			iRow += r
			iCol += c
		end
		return result
	end

	function getMatFromSmiles(allSmiles)
		result = []
		for smiles in allSmiles
			mol = smilestomol(smiles)
			mol = adjacency_matrix(mol.graph)
			push!(result, mol)
		end
		result = adjMat(result)
		return result
	end

	#import the smiles files
	path = joinpath("/Users/punnaamornvivat/Desktop/SURF 2025", "mol.csv");
	german_ref = CSV.read(path, DataFrame)

	#group data into 3 types
	#test used the smaller set of data
	trainData, trainY = german_ref.smiles[1:5], german_ref.lipo[1:5]
	validationData, validationY = german_ref.smiles[71:80], german_ref.lipo[71:80]
	testData, testY = german_ref.smiles[81:end], german_ref.lipo[81:end]

	#create GNNGraph from smiles
	A = getMatFromSmiles(trainData)
	ndata, vector, indicator = verPropTest(trainData)
	B = GNNGraph(A; ndata = ndata, graph_indicator = Int.(indicator)) #I think this line works fine

	######## this is when the code start getting weird
	#this is the function I added. It creates the list of GNNGraphs
	function createGNN(allsmiles)
		result = []
		for sm in allsmiles
			mol = smilestomol(sm)
			gnn = GNNGraph(mol)
		end
		return result
	end

	C = createGNN(trainData)
    	train_loader = zip(C, trainY)
    	test_loader = zip(testData, testY)


	function GNN(din::Int, d::Int, dout::Int)  
   		 GNNChain(GraphConv(din => d), BatchNorm(d), GraphConv(d => d, relu),
       		 Dropout(0.5), Dense(d, dout))
	end
	
	din, d, dout = 3, 4, 1 
	model = GNN(din, d, dout)
	
	function train(graph, train_loader, model)
    		opt = Flux.setup(Adam(0.001), model)
    		for epoch in 1:100
        		for (x, y) in train_loader
            			# x, y = (x, y)
            			grads = Flux.gradient(model) do model
            				ŷ = model(graph, x)
                			Flux.mae(ŷ, y) 
            			end
            			Flux.update!(opt, model, grads[1])
        		end
        
       			if epoch % 10 == 0
            			loss = mean([Flux.mae(model(graph,x), y) for (x, y) in train_loader]) #it crashed here
            			@show epoch, loss
        		end
    		end
    		return model
	end
	
	train(B, train_loader, model)
	
end
