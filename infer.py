import cv2
import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best.pt', help='模型权重路径')
    parser.add_argument('--source', default='armor_video.mp4', help='输入视频路径')
    parser.add_argument('--output', default='result_video.mp4', help='输出视频路径')
    parser.add_argument('--show', action='store_true', help='是否实时显示画面（服务器环境建议不加）')
    args = parser.parse_args()

    # 1. 健壮性检查：模型文件是否存在
    if not os.path.exists(args.model):
        print(f"[Error] 找不到模型文件: {args.model}")
        return

    model = YOLO(args.model)
    
    # 2. 健壮性检查：视频是否能打开
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. 视频写入器配置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    
    print(f"[Info] 开始推理，按 'q' 可退出（如果开启了show）...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[Info] 已处理 {frame_count} 帧...")

        # 推理
        results = model.predict(frame, conf=0.5, stream=False, verbose=False)
        annotated_frame = frame.copy()
        
        for res in results:
            # 绘制检测框、类别、置信度
            annotated_frame = res.plot(img=annotated_frame)
            
            # 提取4个二维关键点并显式绘制
            if res.keypoints is not None and res.keypoints.xy is not None:
                kpts = res.keypoints.xy.cpu().numpy()  # (N, 4, 2)
                for kp in kpts:
                    pts = kp.astype(int)
                    
                    # 可选：过滤掉坐标为(0,0)的无效点
                    if (pts == 0).all():
                        continue
                        
                    # 绘制四角点连线（四边形）
                    for j in range(4):
                        # 确保两个点都有效才连线
                        if not (pts[j] == 0).all() and not (pts[(j+1)%4] == 0).all():
                            cv2.line(annotated_frame, tuple(pts[j]), tuple(pts[(j+1)%4]), (0,255,0), 2)
                            
                    # 绘制关键点圆圈（为了清晰，可覆盖 plot 默认绘制）
                    for pt in pts:
                        if not (pt == 0).all():
                            cv2.circle(annotated_frame, tuple(pt), 5, (0,0,255), -1)
        
        out.write(annotated_frame)
        
        # 4. 使用 try-except 保护 imshow，或通过参数控制
        if args.show:
            try:
                cv2.imshow("Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            except cv2.error:
                print("[Warning] 当前环境不支持 GUI 显示，已自动关闭显示功能。")
                args.show = False # 关闭一次后就禁用，防止反复报错

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[Success] 推理完成，结果已保存至 {args.output}")

if __name__ == "__main__":
    main()
