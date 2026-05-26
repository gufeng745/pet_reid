#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 请根据实际项目调整以下头文件路径 */
#include "alg_type.h"
#include "od_net_desc.h"
#include "alg_object.h"
#include "utils.h"
#include "ipu_yolov8_spec.h"

S32 yolov8_ipu_cpu_postprocess_end2end(APP_ALG_IO_BUF_S *io_buf, od_net_desc_t *net_desc, alg_object_t *objs, U32 *obj_num)
{
    float *buf_pointer = NULL;
    float x1 = 0.0;
    float y1 = 0.0;
    float x2 = 0.0;
    float y2 = 0.0;
    float score = 0.0;
    S32 class_label = 0;
    U32 i = 0;
    U32 j = 0;
    U32 filte_idx = 0;
    U16 accumulate_box_num = 0;

    if (NULL == io_buf || NULL == net_desc || NULL == objs || NULL == obj_num || 0 == *obj_num || NULL == net_desc->spec)
    {
        ALG_ERROR("Invalid args.");
        return ALG_ERR;
    }

    if (ALG_HW_IPU != net_desc->hw)
    {
        ALG_ERROR("hw not ipu.");
        return ALG_ERR;
    }

    if (NET_TYPE_YOLOV8 != net_desc->type)
    {
        ALG_ERROR("Currently not support net type %u cpu postprocess.", net_desc->type);
        return ALG_ERR;
    }

    ipu_yolov8_spec_t *yolov8_desc = NULL;
    yolov8_desc = (ipu_yolov8_spec_t *)net_desc->spec;

    /* end2end模型输出为[1,300,6],每个box: x1 y1 x2 y2 conf cls_id */
    /* 只有一个输出tensor */
    buf_pointer = (float *)(io_buf->out_tensor[0].va);

    /* 遍历300个检测框 */
    for (i = 0; i < 300; i++)
    {
        x1 = buf_pointer[0];
        y1 = buf_pointer[1];
        x2 = buf_pointer[2];
        y2 = buf_pointer[3];
        score = buf_pointer[4];
        class_label = (S32)buf_pointer[5];

        /* 跳过置信度低于阈值的框 */
        if (score < yolov8_desc->score_threshold)
        {
            buf_pointer += 6;
            continue;
        }

        /* 跳过无效框 */
        if ((UTILS_FABS((x1 - x2)) < 1e-6) || (UTILS_FABS((y1 - y2)) < 1e-6))
        {
            buf_pointer += 6;
            continue;
        }

        /* 将结果拷贝到输出数组, 和原逻辑保持一致 */
        if (filte_idx >= *obj_num)
        {
            break;
        }

        for (j = 0; j < net_desc->cls_num; j++)
        {
            if (class_label + 1 == net_desc->cls_info[j].cls_label)
            {
                if (score < net_desc->cls_info[j].conf_thresh)
                {
                    break;
                }

                objs[filte_idx].type = net_desc->cls_info[j].cls_id;
                //confidence
                objs[filte_idx].confidence = score;
                //box coordinate
                objs[filte_idx].bbox.x_min = UTILS_MIN(x1, x2);
                objs[filte_idx].bbox.y_min = UTILS_MIN(y1, y2);
                objs[filte_idx].bbox.x_max = UTILS_MAX(x1, x2);
                objs[filte_idx].bbox.y_max = UTILS_MAX(y1, y2);

                objs[filte_idx].bbox.x_min = CLIP2(objs[filte_idx].bbox.x_min, 0.0, 1.0);
                objs[filte_idx].bbox.y_min = CLIP2(objs[filte_idx].bbox.y_min, 0.0, 1.0);
                objs[filte_idx].bbox.x_max = CLIP2(objs[filte_idx].bbox.x_max, 0.0, 1.0);
                objs[filte_idx].bbox.y_max = CLIP2(objs[filte_idx].bbox.y_max, 0.0, 1.0);
                filte_idx++;
                break;
            }
        }

        buf_pointer += 6;
    }

    *obj_num = filte_idx;

    return ALG_OK;
}
