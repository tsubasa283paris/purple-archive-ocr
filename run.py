"""
コマンドライン引数で指定されたGIFファイルを読み込み、そこに含まれる
文字列情報をJSONとして標準出力に出す。
"""
import argparse
import glob
from typing import List, Union

import cv2
from google.cloud.vision import ImageAnnotatorClient
from google.cloud.vision_v1.types import Feature
from google.oauth2 import service_account


GAPI_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def retrieve_text_from_gif(
        gif_path: str,
        credentials: service_account.Credentials
    ) -> None:
    """
    指定されたGIFファイルのパスから、GIFファイルの構成画像それぞれを
    対象にVision APIを呼び出し、含まれる文字列の一覧を返す。
    """
    # クライアント作成
    client = ImageAnnotatorClient(credentials=credentials)

    # GIF読み込み
    gif = cv2.VideoCapture(gif_path)
    request_param: List[dict] = []
    ret = True
    while ret:
        # 次のフレーム読み込み
        ret, frame = gif.read()
        if not ret:
            break

        # APIリクエスト用のオブジェクト作成
        _, buffer = cv2.imencode(".jpg", frame)
        request_param.append({
            "image": {
                "content": buffer.tobytes()
            },
            "features": [{
                "type_": Feature.Type.DOCUMENT_TEXT_DETECTION
            }],
        })
    
    # APIを呼び出す
    response = client.batch_annotate_images(requests=request_param)
        
    # 結果表示
    for i, air in enumerate(response.responses):
        print("=" * 79)
        print(i)
        print("-" * 79)
        if len(air.text_annotations):
            print(air.text_annotations[0].description)
    
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gif", required=True,
                    help="Path to the GIF file to conduct OCR on")
    ap.add_argument("-c", "--cred",
                    help="Path to the Google Cloud Platform credential JSON "\
                        +"file")
    args = ap.parse_args()

    gif_path: str = args.gif
    cred_path: Union[str, None] = args.cred

    if cred_path is None:
        cred_path_list = glob.glob("./cred/*.json")
        if len(cred_path_list) == 0:
            print("Make sure to put GCP credential JSON file in the 'cred' "\
                  +"directory.")
            exit(1)
        cred_path = cred_path_list[0]

    # 証明書の読み込み
    credentials = service_account.Credentials.from_service_account_file(cred_path)

    # API呼び出しの実施
    retrieve_text_from_gif(gif_path, credentials)
