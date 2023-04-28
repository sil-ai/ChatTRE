from google.cloud import translate


def translate_text(text="Hello, world!"):
    PROJECT_ID = 'studied-flow-385100'
    PARENT = f'projects/{PROJECT_ID}'
    TRANSLATE = translate.TranslationServiceClient()
    data = {
                'contents': [text],
                'parent': PARENT,
                'target_language_code': 'en-US',
            }
    try:
        response = TRANSLATE.translate_text(request=data)
    except TypeError:
        response = TRANSLATE.translate_text(**data)
    return response

if __name__ == '__main__':
    response = translate_text('Wewe ni nani?')