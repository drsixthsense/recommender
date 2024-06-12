sim_threshold = 0.6
if "sim_threshold" in params:
    sim_threshold = params["sim_threshold"] / 100.0
if "profile_sim_threshold" in params:
    profile_sim_threshold = params["profile_sim_threshold"] / 100.0
idx_id_dict, id_idx_dict = get_doc_dicts()
sim_matrix = load_course_sims().to_numpy()
users = []
courses = []
scores = []
res_dict = {}