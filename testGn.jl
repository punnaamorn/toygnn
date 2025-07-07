begin
	using Pkg
	using Graphs
	using MolecularGraph
	using GraphNeuralNetworks
	using RDKitMinimalLib
	using Flux, Graphs, GraphNeuralNetworks
	
	function graphProp(smiles)
		mol = smilestomol(smiles)
		prop = DataStore(molecularWeight = (standard_weight(mol, 1)),
					   HA = (hydrogen_acceptor_count(mol)),
					   HD = (hydrogen_donor_count(mol)))
		return prop
	end

	function verProp(smiles)
		mol = smilestomol(smiles)
		dsVer = DataStore(atomicNumber = (atom_number(mol)),
					   charge = (atom_charge(mol)), 
					   lonePair = (lone_pair(mol)))
		return dsVer
	end

	function getVerVec(smiles)
		mol = smilestomol(smiles)
		vec = [atom_number(mol)',
			   atom_charge(mol)',
			   lone_pair(mol)']
		vec = reduce(vcat, vec)
		return vec
	end

	function edgeProp(smiles)
		mol = smilestomol(smiles)
		dsEdge = DataStore(bondOrder = (bond_order(mol)),
						  aromaticity = (is_aromatic(mol)))
		return dsEdge
	end

	function getGNN(smiles)
		mol = smilestomol(smiles)
		graph = GNNGraph(mol, ndata = verProp(smiles), edata = edgeProp(smiles), gdata=graphProp(smiles))
		dimension = size(graph, 1)
		return graph
	end

	smiles = "O=C1CCCCCN1"
	println(getVerVec(smiles))
	
	using Flux, Graphs, GraphNeuralNetworks

	struct GNN                                # step 1
    	conv1
    	bn
    	conv2
    	dropout
    	dense
	end

	Flux.@layer GNN                         # step 2

	function GNN(din::Int, d::Int, dout::Int) # step 3    
    	GNN(GraphConv(din => d),
        	BatchNorm(d),
        	GraphConv(d => d, relu),
        	Dropout(0.5),
        	Dense(d, dout))
	end

	function (model::GNN)(g::GNNGraph, x)     # step 4
    	x = model.conv1(g, x)
   		x = relu.(model.bn(x))
    	x = model.conv2(g, x)
    	x = model.dropout(x)
    	x = model.dense(x)
    	return x 
	end

	smiles = "O=C1CCCCCN1"
	graph = getGNN(smiles)
	din = 3
	d = 4
	dout = 1
	model = GNN(din, d, dout)                 # step 5
	
	iniVec = getVerVec(smiles)

	y = model(graph, iniVec)  # output size: (dout, g.num_nodes)
	grad = gradient(model -> sum(model(graph, iniVec)), model)

	
end
