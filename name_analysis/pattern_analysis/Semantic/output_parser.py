# eslamxm/mbart-finetune-ar-xlsum mbart:(A,1.0)(O,0.1)(L,0.1),finetune:(F,1.0)(O,0.1)(M,0.1),ar:(L,0.9)(O,0.2)(D,0.2),xlsum:(T,0.9)(O,0.2)(D,0.2)

def parse_text(file_name):
    with open(file_name) as f:
        content = f.read()
    sep_content = content.split('\n')
    for content_segment in sep_content: 
        if content_segment == (''): break
        curr_idx = content_segment.find(' ')
        model_name = content_segment[:curr_idx]
        model_name_suf = model_name.split('/')[-1]
        seg_list = []
        while True:
            curr_idx = content_segment.find(':')
            if curr_idx == -1: break
            seg_list.append(content_segment[content_segment.find('),')+2:curr_idx])
            content_segment = content_segment[curr_idx+1:]
            top_1 = (content_segment[1], float(content_segment[3:6]))
            top_2 = (content_segment[8], float(content_segment[10:13]))
            top_3 = (content_segment[15], float(content_segment[17:20]))
        seg_list = seg_list[1:]
        #print(model_name, seg_list, top_1, top_2, top_3)
        check_content_segment_correctness(model_name_suf, seg_list)

def check_content_segment_correctness(model_name_suf, seg_list):
    for seg in seg_list:
        print(seg)
        model_name_suf.replace(seg, '') # bug
    print(model_name_suf)




            
            
        


parse_text('response_new.txt')
