from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    selected_option = request.form['option']

    # 这里可以根据selected_option进行数据处理
    # 此处简单生成一个示例图
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title(f'Selected Option: {selected_option}')

    # 将图表保存到内存文件中
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
