import pandas as pd
import numpy as np

# Во первых, не будем рассматривать объекты, у которых хотя бы в один день не было кол-во отзывов 116 - это 95.5-квантиль кол-ва отзывов
# Во-вторых, уберем из рассмотрения все дни, в которых кол-во отзывов меньше 15 - 70-квантиль кол-ва отзывов
# Оценки были заменены на 3 группы - 1-3, 4-7, 8-10, так как так проще и отвечает интуитивному пониманию флешмобов.
# Обучив решающее дерево было получено, что самым важным признаком из 9ти рассмотренных является максимальное отношение плохих оценок ко всем среди всех дней. Был получен порог на этот признак threshold = 0.76819924 - если признак больше этого значения, то флешмоб. Также были получены и другие пороги на другие признаки, но важности этих признаков незначительны (Не хочется их добавлять, это возможно переобучение)
# Если не сделать первые 2 пункта - до дни с одной плохой оценкой будут считаться флешмобами
# Подробности в ноутбуке

def flashmob_detector(data):
    new_data = data.copy()
    new_data.vote[new_data.vote==2] = 1
    new_data.vote[new_data.vote==3] = 1
    new_data.vote[new_data.vote==4] = 2
    new_data.vote[new_data.vote==5] = 2
    new_data.vote[new_data.vote==6] = 2
    new_data.vote[new_data.vote==7] = 2
    new_data.vote[new_data.vote==8] = 3
    new_data.vote[new_data.vote==9] = 3
    new_data.vote[new_data.vote==10] = 3
    
    min_border_day = 15
    min_border_obj = 116
    threshold = 0.76819924
    
    answers = []
    
    for obj_id in new_data.obj_id.unique():
        data_obj = new_data[new_data.obj_id == obj_id]
        counts_days = data_obj.pivot_table(index=['day', 'obj_id'], values='vote', aggfunc='count').reset_index()
        if np.max(counts_days.vote) > min_border_obj:
            data_obj_counts = data_obj.merge(counts_days, on=['day', 'obj_id'], how='inner')
            data_obj_counts = data_obj_counts[data_obj_counts.vote_y > min_border_day]
            counts_days = counts_days[counts_days.vote > min_border_day]

            group_1 = data_obj_counts[data_obj_counts.vote_x == 1].pivot_table(index=['day', 'obj_id'], values='vote_x', aggfunc='count').reset_index()
            if(len(group_1) > 0) :
                data_obj_counts = data_obj_counts.merge(group_1, on=['day', 'obj_id'], how='inner')

                counts_days = counts_days.merge(group_1, on=['day', 'obj_id'], how='inner')
                counts_days['rate_1'] = counts_days.vote_x / counts_days.vote

                feature = np.max(counts_days.rate_1)

                if feature > threshold:
                    answers.append([obj_id])
       
    return answers
