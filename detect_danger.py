import cv2
import numpy as np
import onnxruntime as ort
import os
from PIL import ImageFont, ImageDraw, Image

def main():
    # 1. ONNX 모델 파일 경로 설정
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
        return

    # 2. ONNX Runtime 세션 생성
    try:
        session = ort.InferenceSession(model_path)
        print("ONNX 모델 로드 성공!")

        print("입력:")
        for inp in session.get_inputs():
            print(inp.name, inp.shape)

        print("\n출력:")
        for out in session.get_outputs():
            print(out.name, out.shape)

    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 모델 입력 정보
    input_name = session.get_inputs()[0].name

    # 3. 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"

    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 25)
    else:
        font = None

    # 4. 웹캠 시작
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다.")
        return

    print("웹캠 시작. 종료하려면 q를 누르세요.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 화면을 가져올 수 없습니다.")
            break

        # 이미지 전처리
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # ONNX 추론
        outputs = session.run(None, {input_name: img})

        predictions = outputs[0][0]

        best_conf = 0.0
        best_class = "normal"

        class_names = ["normal", "phone", "sleep", "yawn"]

        for pred in predictions:
            confidence = float(pred[4])

            if confidence < 0.3:
                continue

            class_scores = pred[5:]
            class_id = np.argmax(class_scores)

            if confidence > best_conf:
                best_conf = confidence
                best_class = class_names[class_id]

        # 경고 문구 결정
        text_to_show = ""

        if best_class == "phone":
            text_to_show = "[위험] 스마트폰 사용 감지!"

        elif best_class == "sleep":
            text_to_show = "[위험] 졸음운전 감지!"

        elif best_class == "yawn":
            text_to_show = "[주의] 하품 감지!"

        # 한글 출력
        if text_to_show and font:
            img_pil = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            draw = ImageDraw.Draw(img_pil)

            draw.text(
                (20, 40),
                text_to_show,
                font=font,
                fill=(255, 0, 0)
            )

            frame = cv2.cvtColor(
                np.array(img_pil),
                cv2.COLOR_RGB2BGR
            )

        elif text_to_show:
            cv2.putText(
                frame,
                "[WARNING] Danger Detected!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # 화면 출력
        cv2.imshow(
            "Driver Hazard Detection Test",
            frame
        )

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 처리
    cap.release()
    cv2.destroyAllWindows()

    print("프로그램을 안전하게 종료했습니다.")

if __name__ == "__main__":
    main()