/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;


/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithInterpreter";

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private ByteBuffer imgData;

  private MappedByteBuffer tfLiteModel;
  private Interpreter.Options tfLiteOptions;
  private Interpreter tfLite;

  private float[][] ageMap;
  private float[][] genderMap;
  private static final int AGE_SIZE = 4;
  private static final int GENDER_SIZE = 2;
  private static final float WHITE_THRESH = 255f;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    AssetManager am = context.getAssets();
    InputStream is = am.open(labelFilename);

    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
    MetadataExtractor metadata = new MetadataExtractor(modelFile);
    try (BufferedReader br =
                 new BufferedReader(
                         new InputStreamReader(is))) {
      String line;
      while ((line = br.readLine()) != null) {
        //Log.w(TAG, line);
        d.labels.add(line);
      }
    }

    d.inputSize = inputSize;

    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      options.setUseXNNPACK(true);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  @Override
  public String recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];


        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model

          // divide image pixels by 255 to get values in range from 0 to 1

//          imgData.putFloat(((pixelValue >> 16) & 0xFF) / WHITE_THRESH);
//          imgData.putFloat(((pixelValue >> 8) & 0xFF) / WHITE_THRESH);
//          imgData.putFloat((pixelValue & 0xFF)  / WHITE_THRESH);

          //doing the color change here from RGB to BGR

          imgData.putFloat((pixelValue & 0xFF) / WHITE_THRESH);
          imgData.putFloat(((pixelValue >> 8) & 0xFF) / WHITE_THRESH);
          imgData.putFloat(((pixelValue >> 16) & 0xFF) / WHITE_THRESH);
        }
      }
    }

    // convert RGB to BGR

    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    ageMap = new float[1][AGE_SIZE];
    genderMap = new float[1][GENDER_SIZE];

    outputMap.put(0, ageMap);
    outputMap.put(1, genderMap);


    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");


    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


    Log.w("AGE", "!!!AGE: " + ageMap[0][0] + "|" + ageMap[0][1] + "|" + ageMap[0][2] + "|" + ageMap[0][3]);
    Log.w("GENDER", "!!!GENDER: " + genderMap[0][0] + "|" + genderMap[0][1]);

    Integer ageInd = 0;
    Integer genderInd = 0;

    Float max = -1f;
    for (int i = 0; i < ageMap[0].length; i++) {
      float currValue = ageMap[0][i];

      if (currValue > max) {
        max = currValue;
        ageInd = i;
      }

    }

    max = -1f;
    for (int i = 0; i < genderMap[0].length; i++) {
      float currValue = genderMap[0][i];

      if (currValue > max) {
        max = currValue;
        genderInd = i;

      }

    }


    //LOGGER.i("IND AGE: " + ageInd + " | IND Gender: " + genderInd);

    List<String> genderList = Arrays.asList("Female", "Male");
    List<String> ageList = Arrays.asList("0-14yo", "15-40yo", "41-60yo", "61-100yo");


    String label = "AGE: " + ageList.get(ageInd) + " | Gender: " + genderList.get(genderInd);


    Trace.endSection();

    final int numDetectionsOutput = 1;
    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    //Recognition rec = new Recognition("0", label, new RectF());

    //recognitions.add( rec );


    Trace.endSection();
    return label;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (tfLite != null) {
      tfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) {
      tfLiteOptions.setUseNNAPI(isChecked);
      recreateInterpreter();
    }
  }

  private void recreateInterpreter() {
    tfLite.close();
    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
  }
}
