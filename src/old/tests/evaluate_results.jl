using Plots

result_path = "./src/tests/results/"
result_prefix = "fullres_"
partresult_prefix = "partialres_"

molecule = "graphene"
#"graphene", "H2", "silicon", "GaAs", "TiO2"
methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "scf_naive", "rcg_ea", "rcg_inea_shift", "rcg_ea_ah_shift", "scf"]
