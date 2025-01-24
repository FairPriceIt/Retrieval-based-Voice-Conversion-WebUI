from flask import Flask, request, jsonify, send_file
import os
from scipy.io import wavfile
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from configs.config import Config
from infer.modules.vc.modules import VC
import torch
import numpy as np
import librosa

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
ALLOWED_EXTENSIONS = {'wav'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize Config
config = Config()
config.device = 'cpu'
config.is_half = False

# Initialize female VC and index
female_vc = VC(config)
female_vc.get_vc('splicegirl_e120_s18480.pth')
female_index_path = '/root/Retrieval-based-Voice-Conversion-WebUI/logs/splicegirl_e120_s18480/added_IVF3705_Flat_nprobe_1_splicegirl_v2.index'

# Initialize male VC and index
male_vc = VC(config)
male_vc.get_vc('narpy.pth')
male_index_path = '/root/Retrieval-based-Voice-Conversion-WebUI/logs/narpy/added_IVF811_Flat_nprobe_1_narpy_v2.index'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_pitch(input_path):
    # Load audio file
    y, sr = librosa.load(input_path, sr=None)

    # Limit frequency range to human vocal range
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=85.0, fmax=300.0)

    # Filter valid pitches and calculate mean
    valid_pitches = [np.mean(p[(p > 85) & (p < 300)]) for p in pitches.T if len(p[p > 0]) > 0]
    mean_pitch = np.nanmean(valid_pitches) if valid_pitches else 0

    # Define pitch thresholds
    FEMALE_PITCH_THRESHOLD = 165  # Female lower limit
    MALE_PITCH_THRESHOLD = 145    # Male upper limit

    print(f"Valid Pitches: {valid_pitches}")
    print(f"Mean Pitch: {mean_pitch}")  # Debugging output

    # Classify based on pitch
    if mean_pitch >= FEMALE_PITCH_THRESHOLD:
        return "female"
    elif mean_pitch <= MALE_PITCH_THRESHOLD:
        return "male"
    else:
        return "unknown"

voiceCounter = 0

def process_voice_clone(input_path, output_path, args):
    # Analyze pitch of the input
    global voiceCounter
    voiceCounter += 1
    pitch_category = analyze_pitch(input_path)

    # Select VC and index path based on pitch
    if voiceCounter%2 ==0:
        vc = female_vc
        index_path = female_index_path
    else:
        vc = male_vc
        index_path = male_index_path

    # Generate voice clone
    try:
        _, wav_opt = vc.vc_single(
            0,  # Placeholder for speaker ID
            input_path,
            args.get('f0up_key', 0),
            None,  # Placeholder for additional features
            'rmvpe',  # Use RMVPE as pitch extraction method
            index_path,
            None,
            args.get('index_rate', 0.66),
            args.get('filter_radius', 3),
            args.get('resample_sr', 0),
            args.get('rms_mix_rate', 1),
            args.get('protect', 0.33),
        )
    except Exception as e:
        raise RuntimeError(f"Error during voice conversion: {str(e)}")

    # Save the output WAV file
    wavfile.write(output_path, wav_opt[0], wav_opt[1])

@app.route('/clone', methods=['POST'])
def clone_voice():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Extract additional parameters from the form
        args = {
            'f0up_key': int(request.form.get('f0up_key', 0)),
            'index_rate': float(request.form.get('index_rate', 0.66)),
            'filter_radius': int(request.form.get('filter_radius', 3)),
            'resample_sr': int(request.form.get('resample_sr', 0)),
            'rms_mix_rate': float(request.form.get('rms_mix_rate', 1)),
            'protect': float(request.form.get('protect', 0.33)),
        }

        # Process the voice cloning
        try:
            process_voice_clone(input_path, output_path, args)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return send_file(output_path, as_attachment=True)
    else:
        return jsonify({'error': 'Invalid file type. Only WAV files are allowed.'}), 400


if __name__ == "__main__":
    app.run(debug=False)
