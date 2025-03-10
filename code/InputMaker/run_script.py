from InputMaker import InputMaker

dataset_directory_dict ={
"L1U":"BPIX_L1U_template_events_d21601_d21800",
"L1F":"BPIX_L1F_template_events_d21901_d22100",
"L2old":"template_events_d22501_d22700",
"L2new":"template_events_d22201_d22400",
"L3m":"BPIX_L3m_template_events_d83201_d83400",
"L3p":"BPIX_L3p_template_events_d82901_d83100",
"L4m":"BPIX_L4m_template_events_d78501_d78700",
"L4p":"BPIX_L4p_template_events_d77001_d77200"
}

template_id_dict={
    "L1F":1122,
    "L1U":1123,
    "L3m":1125,
    "L3p":1126,
    "L4m":1127,
    "L4p":1128
}

decapitation=True
for layer in ["L1F","L1U","L3m","L3p","L4m","L4p"]:
    file_tag = dataset_directory_dict[layer]
    input_file = f"/ssd-data1/mrogul/orig_clusters/{file_tag}"
    output_folder = "/ssd-data1/mrogul/clusters/v2"
    template_id = template_id_dict[layer]
    input_maker = InputMaker(input_file,output_folder,layer,template_id,decapitation=decapitation)
    #input_maker.unzip_dir()
    input_maker.convert_txt_files()    
    #input_maker.clear()