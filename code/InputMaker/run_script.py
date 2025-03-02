from InputMaker import InputMaker

dataset_directory_dict ={
"L1U":"template_events_d21601_d21800",
"L1F":"BPIX_L1F_template_events_d21901_d22100",
"L2old":"template_events_d22501_d22700",
"L2new":"template_events_d22201_d22400",
"L3m":"template_events_d83201_d83400",
"L3p":"template_events_d82901_d83100",
"L4m":"template_events_d78501_d78700",
"L4p":"template_events_d77001_d77200"
}

template_id_dict={
    "L1F":1122
}

for layer in ["L1F"]:
    base_folder = "/ssd-data1/mrogul/orig_clusters"
    layer_folder = dataset_directory_dict[layer]
    input_folder = f"{base_folder}/{layer_folder}/"
    output_folder = "/ssd-data1/mrogul/clusters/v2"
    template_id = template_id_dict[layer]
    input_maker = InputMaker(input_folder,output_folder,layer,template_id)
    #input_maker.unzip_dir()
    input_maker.convert_txt_files()    
    #input_maker.clear()