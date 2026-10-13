import cv2
import argparse
from ultralytics import YOLO

def main():
	parse = argparse.ArgumentParser()
	parser.add_argument('--model',default='best.pt',help='模型权重路径')
	parser.add_argument('--source',default='armor_video.mp4',help='输入视频路径')
	parser.add_argument('--output',default='result_vedio.mp4',help='输出视频路径')
	args = parser.parse_args()
	
	model = YOLO(args.model)
	cap = cv2.VedioCapture(args.source)
	fps = cap.get(cv2.CAP_PROP_FPS)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(args.output,fourcc,fps,(w,h))
	
	while cap.isOpened():
		ret,frame = cap.read()
		if not ret:break
		
		results = model.predict(frame,conf=0.5,stream=False)
		annotated_frame = frame.copy()
		
		for res in results:
			# 绘制默认检测框、类别、置信度
			annotated_frame = res.plot(img=annotated_frame)
			
			#提取4个二维关键点并显示绘制（确保关键点清晰可见）
			if res.keypoints is not None:
				kpts = res.keypoints.xy.cpu().numpy()  #(N,4,2)
				for kp in kpts:
					pts = kp.astype(int)
					#绘制四角点连线（四边形）
					for j in range(4):
						cv2.line(annotated_frame,tuple(pts[j]),tuple(pts[(j+1)]%4),(0,255,0),2)
						#绘制关键点圆圈
						for pt in pts: 
							cv2.circle(annotated_frame,tuple(pt),5,(0,0,255),-1)
		out.write(annotated_frame)
		cv2.imshow("Inference",annotated_frame)
		if cv2.waitKey(1) & 0xFF ==	ord('q'): break
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	print(f"结果已保存至{args.output}")
if _name_ == "_main_":
   main()				
