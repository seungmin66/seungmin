import cv2
import numpy as np
import onnxruntime as ort
import os
import winsound  # 윈도우 기본 경고음 시스템 내장 라이브러리
import pyttsx3   # 텍스트를 음성으로 읽어주는 라이브러리
import threading # 🚨 소리 재생 중 화면과 카운터가 멈추는 현상을 막기 위한 라이브러리
from PIL import ImageFont, ImageDraw, Image

# 음성 안내(TTS)를 화면 멈춤 없이 백그라운드에서 실행하는 함수
def speak_async(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# 윈도우 비프음을 백그라운드에서 실행하는 함수
def beep_async(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# 윈도우 시스템 알림음을 백그라운드에서 실행하는 함수
def message_beep_async(type):
    threading.Thread(target=winsound.MessageBeep, args=(type,), daemon=True).start()

def main():
    # 1. ONNX 모델 파일 경로 설정
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        print(f"[에러] '{model_path}' 파일을 찾을 수 없습니다. OSP 폴더 안으로 이동 후 실행하세요.")
        return

    # 2. ONNX Runtime 세션 생성
    try:
        session = ort.InferenceSession(model_path)
        print("[정보] ONNX 모델 로드 성공!")
    except Exception as e:
        print(f"[에러] 모델 로드 실패: {e}")
        return

    # 모델 입력 정보
    input_name = session.get_inputs()[0].name

    # 3. 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else None

    # 4. 웹캠 시작
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[에러] 카메라를 열 수 없습니다.")
        return

    print("[정보] 웹캠 시작. 종료하려면 q를 누르세요.")

    # 연속 프레임 카운터 변수 (지속 시간 체크용)
    sleep_counter = 0
    phone_counter = 0
    
    # 소리가 무한 중복으로 겹쳐서 프로그램이 튕기는 것을 막기 위한 주기 변수
    audio_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[에러] 카메라 화면을 가져올 수 없습니다.")
            break

        # 웹캠 좌우반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        h_orig, w_orig = frame.shape[:2]

        # 이미지 전처리 (640x640 크기 맞춤)
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

        # 객체 검출 및 사각형 박스 치기
        for pred in predictions:
            confidence = float(pred[4])
            
            # 🚨 [수정] 박스가 너무 많이 나오는 현상 방지를 위해 기준을 0.4에서 0.6으로 상향 조정
            if confidence < 0.6:
                continue

            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            current_class = class_names[class_id]

            cx, cy, w_box, h_box = pred[:4]
            x1 = int((cx - w_box / 2) * w_orig / 640)
            y1 = int((cy - h_box / 2) * h_orig / 640)
            x2 = int((cx + w_box / 2) * w_orig / 640)
            y2 = int((cy + h_box / 2) * h_orig / 640)

            if current_class == "phone":
                box_color = (0, 0, 255)
            elif current_class == "sleep":
                box_color = (255, 255, 0)
            elif current_class == "yawn":
                box_color = (0, 165, 255)
            else:
                box_color = (0, 255, 0)

            if current_class != "normal":
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if confidence > best_conf:
                best_conf = confidence
                best_class = current_class

        # 지속 시간 카운트 계산
        if best_class == "sleep":
            sleep_counter += 1
        else:
            sleep_counter = max(0, sleep_counter - 1)

        if best_class == "phone":
            phone_counter += 1
        else:
            phone_counter = max(0, phone_counter - 1)

        # 시스템 기본 문구 및 상태 초기화
        text_to_show = "[상태] 정상 운전 중"
        text_color = (0, 255, 0) # 초록색
        
        # 최대 누적 위험 수치 선택
        current_hazard_score = max(sleep_counter, phone_counter)

        # 하품 감지 시 기본 주의 처리
        if best_class == "yawn" and current_hazard_score < 15:
            text_to_show = "1단계: [주의] 하품 감지 - 화면 알림 수신"
            text_color = (0, 165, 255) # 주황색

        # -------------------------------------------------------------
        # 위험도 단계별 조건문 및 백그라운드 사운드 연동 (딜레이 완전 제거)
        # -------------------------------------------------------------
        
        # 3단계: 초고위험 상태 판단 시 즉각적이고 강한 사이렌 경고음 및 위험 알림 음성 출력
        # 위험 행동이 약 2.5초 이상 장시간 지속될 때 발생
        if current_hazard_score >= 75:
            text_to_show = "3단계: [위험] 초고위험 상태! 즉시 위험지역 대피 및 강한 경고음 발생"
            text_color = (0, 0, 255) # 빨간색
            
            if audio_cooldown == 0:
                # 🚨 [수정] 백그라운드 스레드로 비동기 재생하여 화면 및 카운터 정지 버그 해결
                beep_async(2500, 400)
                speak_async("위험 상태입니다. 즉시 행동을 중단하고 안전을 확보하세요.")
                audio_cooldown = 90 # 알림 간격 제어용 쿨타임

        # 2단계: 음성 알림 안내 (상태가 지속될 경우 운전자 주의 유도)
        # 위험 행동이 약 1초~2.5초 사이로 지속되는 단계
        elif current_hazard_score >= 30:
            text_to_show = "2단계: [경고] 위험 행동 지속 - 음성 안내 방송 중"
            text_color = (0, 69, 255) # 짙은 주황색
            
            if audio_cooldown == 0:
                if best_class == "sleep":
                    speak_async("졸음이 감지되었습니다. 전방을 주시하십시오.")
                elif best_class == "phone":
                    speak_async("운전 중 휴대폰 사용은 위험합니다.")
                audio_cooldown = 90

        # 1단계: 화면 알림 및 간단한 알림 소리 
        # 위험 행동의 전조 혹은 짧은 발생 시점 (약 0.3초~1초)
        elif current_hazard_score >= 10:
            text_to_show = "1단계: [알림] 위험 행동 감지 - 화면 경고 메시지"
            text_color = (0, 165, 255) # 주황색
            
            if audio_cooldown == 0:
                message_beep_async(winsound.MB_ICONEXCLAMATION)
                audio_cooldown = 45

        # 오디오 출력 오버플로우 방지용 쿨타임 카운터 다운
        if audio_cooldown > 0:
            audio_cooldown -= 1

        # 한글 안 깨지게 경고 문구 출력
        if font:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            fill_color = (text_color[2], text_color[1], text_color[0])
            draw.text((20, 40), text_to_show, font=font, fill=fill_color)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(frame, text_to_show, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # 최종 화면 출력
        cv2.imshow("Driver Hazard Detection Test", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 처리
    cap.release()
    cv2.destroyAllWindows()
    print("[정보] 프로그램을 안전하게 종료했습니다.")

if __name__ == "__main__":
    main()