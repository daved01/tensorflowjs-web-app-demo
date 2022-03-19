import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './app.css';
import exampleImage from './image.jpg';


async function runModel() {
    const model = await tf.loadGraphModel('https://raw.githubusercontent.com/daved01/tensorflowjs-web-app-demo/main/models/fullyConvolutionalModelTfjs/model.json');

    // Get content image
    let image = new Image(256,256);
    image.src = exampleImage;

    // Convert content image to tensor and add batch dimension
    let tfTensor = tf.browser.fromPixels(image); 
    
    tfTensor = tfTensor.div(255.0);
    tfTensor = tfTensor.expandDims(0);
    tfTensor = tfTensor.cast("float32");
    
    // Run image through model
    const pred = model.predict(tfTensor);
    
    // Convert tensor to image
    let outputTensor = pred.squeeze();
    
    // Scale to range [0,1] from [-1,1]
    outputTensor = outputTensor.mul(0.5);
    outputTensor = outputTensor.add(0.5);

    // Prepare rendering of the result
    const canvas = document.getElementById('canvas'); 
    await tf.browser.toPixels(outputTensor, canvas);      
}


function App(props) {  
    return (
        <div className="main">
            <h1>App</h1>
            <div className="imageContainer">
                <img className="myImage" src={exampleImage} alt="Image" height={256} width={256} />
                <canvas className="myImage" id="canvas" width={256} height={256}> </canvas>
            </div>  
            <div className="myButtonPos">
                <button className="myButton" onClick={runModel}>Run model</button>
            </div>                              
        </div>
    );
}

export default App;