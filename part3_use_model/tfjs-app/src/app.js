import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './app.css';

import exampleImage from './image.jpg';



class App extends React.Component {
    runModel = async () => {
        try {
            const model = await tf.loadGraphModel('https://raw.githubusercontent.com/daved01/test-models/master/mediumNetDeconvs/model.json');

            // Get content image
            let image = new Image(256,256);
            image.src = exampleImage;
        
            // Convert content image to tensor and add batch dimension
            let tfTensor = tf.browser.fromPixels(image); 
            
            tfTensor = tfTensor.div(255.0);
            tfTensor = tfTensor.expandDims(0);
            tfTensor = tfTensor.cast("float32");
            
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
        catch(e) {
            console.log(e);
        }
    }


    render () {
        return (
            <div className="main">
                <h1 className="heading" >App</h1>
                <div className="imageContainer">
                    <img className="myInputImage" src={exampleImage} alt="Image" height={256} width={256} />
                    <canvas className="myOutputImage" id="canvas" ref="canvas" width={256} height={256}> </canvas> 
                </div>  
                <div className="buttonPos">
                    <button className="myButton" onClick={this.runModel} >Run model</button>
                </div>                       
            </div>
        );
    }
}

export default App;