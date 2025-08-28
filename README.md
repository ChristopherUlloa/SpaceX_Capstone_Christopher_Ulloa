#proyecto christopher ulloa python ibm

Instalar el entorno virtual: python -m venv .venv

Activar el entorno: .venv\Scripts\activate

Instalar dependencias: pip install -r requirements.txt

Instalar Kaleido (para Visual Studio Code): pip install kaleido

Instalar Chrome (para Visual Studio Code): python -c "import kaleido; print('Downloading Chrome...'); p=kaleido.get_chrome_sync(); print('Chrome at:', p)"

Ejecutar el script principal: python scripts/generate_assets.py
