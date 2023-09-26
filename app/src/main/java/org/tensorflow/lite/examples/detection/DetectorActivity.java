/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import com.androidnetworking.AndroidNetworking;
import com.androidnetworking.common.Priority;
import com.androidnetworking.error.ANError;
import com.androidnetworking.interfaces.JSONObjectRequestListener;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {

  int errorCode = 400;

  private static final String LOCAL_DATA_FILE = "local_data.json";
  private static final Logger LOGGER = new Logger();
  private int faceCount = 0;
  private Set<Integer> recognizedFaceIds = new HashSet<>();

  // Configuration values for the prepackaged SSD model.
  // AgeGenModel
  private static final int TF_OD_API_INPUT_SIZE = 80;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "face_model_v5.tflite";

  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(720, 1280);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap portraitBmp = null;
  private Bitmap faceBmp = null;

  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // Face detector
  FaceDetector faceDetector;
  private String TAG = "DEMOGRAPHICS";


  private List<Face> trackedFaces = new ArrayList<>();
  private Set<Integer> uniqueFaceIds = new HashSet<>();

  private Handler handler = new Handler();
  private long lastFaceDetectionTime = 0;
  private String formattedFaceCountStartTime;
  private String formattedFaceCountEndTime;
  private Date faceCountEndTime;
  private int liveFaceCount = 0; // For live faces
  private int totalFaceCount = 0; // For total faces
  int tempPosition = 0;
  int position = 0;
  List<DemographicsData.DataInterval> dataIntervalList;
  DemographicsData jsonData;
  String android_id = "";

  Handler newHandler = new Handler();
  Runnable runnable;

  String data = "";
  String tempData = "";



  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);


    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                    .setContourMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                    .enableTracking()
                    .build();


    faceDetector = FaceDetection.getClient(options);

    faceCountTextView = findViewById(R.id.face_count_text_view);

    try {
      String android_id = "ee5d9c851e49058d";
      //+ettings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);
    } catch (Exception e) {
      e.printStackTrace();

    }
    jsonData = new DemographicsData();
    dataIntervalList = new ArrayList<>();

    try {
      checkAndSendOfflineData();
    } catch (IOException | JSONException e) {
      throw new RuntimeException(e);
    }

  }

  private void checkAndSendOfflineData() throws IOException, JSONException {

    File localFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), LOCAL_DATA_FILE);

    Log.e(TAG, "File Path: "+localFile.getAbsolutePath());

    if (localFile.exists()) {
      // Create a StringBuilder to hold the JSON string
      StringBuilder jsonString = new StringBuilder();

      // Create a BufferedReader to read the JSON file
      BufferedReader reader = new BufferedReader(new FileReader(localFile));

      // Read the file line by line and append it to the StringBuilder
      String line;
      while ((line = reader.readLine()) != null) {
        jsonString.append(line);
      }

      // Close the BufferedReader
      reader.close();

      // Create a JSONObject from the JSON string
      JSONObject jsonObject = new JSONObject(jsonString.toString());


      data = String.valueOf(jsonString);

      // Now you have a JSONObject representing the JSON data from the file
      System.out.println(jsonObject.toString(4));
    }

  }



  private void runTask(int size) {

    runnable = () -> {
      // Your task code goes here
      // This code will run every 5 seconds
      // Replace this with your actual task logic
      // For example, update UI elements, fetch data, etc.
      // Calculate the time elapsed since the last face detection
      long currentTime = System.currentTimeMillis();
      // Get current date
      faceCountEndTime = new Date();

      // Define the date format
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");

      // Format the date
      formattedFaceCountEndTime = dateFormat.format(faceCountEndTime);


      long elapsedTime = currentTime - lastFaceDetectionTime;

      // If no faces have been detected in the last 5 seconds, add live face count to total count

      /*for (Face trackedFace : trackedFaces) {
        if (trackedFace.getTrackingId() != uni)
      }*/
      totalUniqueFaceCount = size;

        totalFaceCount += liveFaceCount;

        // Reset live face count
        liveFaceCount = 0;

        // Update the UI or perform any necessary actions with the total face count
        updateFaceCountUI();
        try {

          new AsyncTask<Void, Void, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {
              return isInternetConnected(getApplicationContext());
            }

            @Override
            protected void onPostExecute(Boolean isConnected) {
              if (isConnected) {
                // Internet is available, so send the data to the server
                tempData = saveDemographicsDataTemp();
                if (data!= null && data != ""){
                  tempData = data;

                }
                sendDemographicStatsToServer(tempData);
                //sendDemographicStatsToServer(tempData);

                data = "";
                tempData = "";

              } else {
                saveDemographicsDataLocally();
                position++;
                // Handle the case where there is no internet connection
                // You can show an error message to the user or take any other appropriate action.
              }
            }
          }.execute();

        } catch (Exception e) {
          // Handle exceptions here if necessary
          throw new RuntimeException(e);
        }



      // After the task is complete, post the same Runnable again
      //newHandler.postDelayed(this, 5000); // 5000 milliseconds = 5 seconds
    };

    // Post the initial Runnable to start the task
    newHandler.postDelayed(runnable, 10000); // Start in 10 sec

  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);


    try {
      detector = TFLiteObjectDetectionAPIModel.create(
                      this,
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();



    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    LOGGER.i("CropW, cropH"+cropW+" "+cropH);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);


    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);


    Matrix frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    targetW, targetH,
                    sensorOrientation, MAINTAIN_ASPECT);



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            canvas -> {
              tracker.draw(canvas);
              if (isDebug()) {
                tracker.drawDebug(canvas);
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  public void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");


    //CODE that gets the pixels from the camera
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }


    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);


    faceDetector
            .process(image)
            .addOnSuccessListener(faces -> {
              if (faces.size() == 0) {
                updateResults(currTimestamp, new LinkedList<>());
                return;
              }
              runInBackground(
                      () -> onFacesDetected(currTimestamp, faces));
            });


  }

  // Face Processing
  private Matrix createTransform(final int srcWidth, final int srcHeight, final int dstWidth, final int dstHeight, final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

    if (applyRotation != 0) {

      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;

  }

  private String saveDemographicsDataTemp() {

    String data= "";
    DemographicsData.DataInterval dataInterval= new DemographicsData.DataInterval();
    dataInterval.setStartTime(formattedFaceCountStartTime);
    dataInterval.setEndTime(formattedFaceCountEndTime);
    dataInterval.setPeopleCount(trackedFaces.size());
    //dataInterval.setTrackingId(trackedFaces.get(0).getTrackingId());
    dataIntervalList.add(tempPosition, dataInterval);
    jsonData.setHardwareKey("b677c00e1cd704ef");
    jsonData.setData(dataIntervalList);
    Gson gson = new Gson();
    Log.e(TAG, "Temp data:"+gson.toJson(jsonData));

    return gson.toJson(jsonData);

  }

  private String saveDemographicsDataLocally() {
    ObjectMapper objectMapper = new ObjectMapper();

    data = "";
    DemographicsData.DataInterval dataInterval = new DemographicsData.DataInterval();

    dataInterval.setStartTime(formattedFaceCountStartTime);
    dataInterval.setEndTime(formattedFaceCountEndTime);
    dataInterval.setPeopleCount(trackedFaces.size());
    //dataInterval.setTrackingId(trackedFaces.get(0).getTrackingId());
    dataIntervalList.add(position, dataInterval);

    //position++;



    //DemographicsData jsonData = new DemographicsData();
    jsonData.setHardwareKey("b677c00e1cd704ef");
    jsonData.setData(dataIntervalList);
    //Log.e(TAG, "Tracking ID "+dataIntervalList.get(0).getTrackingId());

    try {
      File documentsDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
      if (!documentsDirectory.exists()) {
        documentsDirectory.mkdirs(); // Create the directory if it doesn't exist.
      }
      File localFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), LOCAL_DATA_FILE);

      Log.e(TAG, "File Path: "+localFile.getAbsolutePath());

        if (localFile.exists()) {
          Reader fileReader = null;
          fileReader = new FileReader(localFile);



          FileInputStream inputStream = new FileInputStream(localFile);
          InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
          BufferedReader reader = new BufferedReader(inputStreamReader);
          StringBuilder existingDataBuilder = new StringBuilder();
          String line;
          while ((line = reader.readLine()) != null) {
            existingDataBuilder.append(line);
          }
          reader.close();

          String existingDataString = existingDataBuilder.toString();
          Log.e(TAG, "existingDataString: "+existingDataString);

          // Parse the existing data into a Java object (if applicable)
          DemographicsData existingJsonData = objectMapper.readValue(localFile, DemographicsData.class);

          existingJsonData.getData().addAll(dataIntervalList);

          // Write the updated data back to the file
          objectMapper.writeValue(localFile, existingJsonData);
          data = objectMapper.writeValueAsString(existingJsonData);
          Log.e(TAG, "Data updated: "+data);
          Toast.makeText(this, "Updated in file", Toast.LENGTH_SHORT).show();
      }else {

          objectMapper.writeValue(localFile, jsonData);
          data = objectMapper.writeValueAsString(jsonData);
          Toast.makeText(this, "Created the new file", Toast.LENGTH_SHORT).show();
      }

      //Log.e(TAG, "File doesn't exist: "+data);
      /*try {
        localFile.createNewFile(); // Create the file.
        // You can now write data to this file.
      } catch (IOException e) {
        e.printStackTrace();
      }*/


    } catch (Exception e) {
      Log.e("Data Exception: ", e.toString());
    }

    //dataIntervalList.clear();

    return data;

  }

  public static boolean isInternetConnected(Context context) {
    ConnectivityManager connectivityManager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
    if (connectivityManager != null) {
      NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
      if (activeNetworkInfo != null && activeNetworkInfo.isConnectedOrConnecting()) {
        try {
          HttpURLConnection urlc = (HttpURLConnection) (new URL("https://8.8.8.8").openConnection());
          urlc.setRequestProperty("User-Agent", "Test");
          urlc.setRequestProperty("Connection", "close");
          urlc.setConnectTimeout(2000);
          urlc.connect();
          return (urlc.getResponseCode() >= 200 && urlc.getResponseCode() < 300);
        } catch (IOException e) {
          Log.e("SignageDisplay", ""+e);
        }

        return false;
      }
    }
    return false;
  }


  private void onFacesDetected(long currTimestamp, List<Face> faces) {

    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    final Canvas canvas = new Canvas(cropCopyBitmap);
    final Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);

    final List<Detector.Recognition> mappedRecognitions = new LinkedList<>();


    //final List<Classifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);

    final Canvas cvFace = new Canvas(faceBmp);

    // Update trackedFaces based on newly detected faces

    updateTrackedFaces(faces);



    for (Face face : faces) {
      //.e("FaceRecognition", "Bounding Box: " + face.getBoundingBox());
      //Log.e("FaceRecognition", "Face Landmarks: " + face.getAllLandmarks());
      Integer trackingId = face.getTrackingId();
      //Log.e("FaceRecognition", "Tracking ID: " + trackingId);
      if(trackingId!=null){


        int faceId = trackingId;
        Log.e("FaceRecognition", "Face ID: " + faceId);

        // Check if the face is already recognized
        if (!recognizedFaceIds.contains(faceId)) {
          recognizedFaceIds.add(faceId);
          faceCount++;
          Log.e("FaceRecognition", "New face detected. Face Count: " + faceCount);
        }
      }

       // Unique identifier for the face



      LOGGER.i("FACE" + face.toString());
      LOGGER.i("Running detection on face " + currTimestamp);
      //results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());

      //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {


        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);


        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        if ((int)faceBB.left > 0 && (int)faceBB.top > 0 && ((int)faceBB.left + (int)faceBB.width()) < portraitBmp.getWidth()
                && ((int)faceBB.top + (int)faceBB.height()) < portraitBmp.getHeight() ) {

          crop = Bitmap.createBitmap(portraitBmp, //prev portraitBmp
                  (int) faceBB.left,
                  (int) faceBB.top,
                  (int) faceBB.width(),
                  (int) faceBB.height());
          crop = Bitmap.createScaledBitmap(crop, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, true);

          final long startTime = SystemClock.uptimeMillis();
          // Passing the bitmap image into the TensorFlow Lite Object Detection API recognize image function
          final String resultLabel = detector.recognizeImage(crop); //prev faceBmp
          lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
          final Detector.Recognition result = new Detector.Recognition("0", resultLabel, confidence, boundingBox);
          result.setLocation(boundingBox);
          mappedRecognitions.add(result);

        }

      }


    }

    updateFaceCountUI();
    //Log.e("FaceCount " ,""+ faceCount);
    updateResults(currTimestamp, mappedRecognitions);

  }

  private final Set<Integer> uniqueTrackingIDs = new HashSet<>();
  private int totalUniqueFaceCount = 0;
  long lastUpdateTime = System.currentTimeMillis();

  private void updateTrackedFaces(List<Face> detectedFaces) {

    try {

      // Update trackedFaces with newly detected faces
      List<Face> updatedTrackedFaces = new ArrayList<>();

      if (detectedFaces.isEmpty()) {
        // No faces detected in the current frame, clear the tracked faces
        trackedFaces.clear();

      } else {
        Date faceCountStartTime = new Date();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");

        long currentTime = System.currentTimeMillis();

        // Check if it's been 5 seconds since the last update
      /*if (currentTime - lastUpdateTime >= 5000) {
        // Update the total count with the count of unique tracking IDs
        totalUniqueFaceCount = uniqueTrackingIDs.size();

        // Update the last update time
        lastUpdateTime = currentTime;
      }*/

        for (Face detectedFace : detectedFaces) {
          boolean isSimilar = false;

          // Compare detectedFace with trackedFaces
          for (Face trackedFace : trackedFaces) {
            Integer trackingIdWrapper = detectedFace.getTrackingId();
            int trackingID = trackingIdWrapper;

            if (!uniqueTrackingIDs.contains(trackingID)) {
              // Add the tracking ID to the set
              uniqueTrackingIDs.add(trackingID);
              formattedFaceCountStartTime = dateFormat.format(faceCountStartTime);
              runTask(uniqueTrackingIDs.size());
            }
            float similarity = calculateSimilarityWithMotion(detectedFace, trackedFace);

            // Adjust the SIMILARITY_THRESHOLD as needed
            float SIMILARITY_THRESHOLD = 0.5f;

            if (similarity >= SIMILARITY_THRESHOLD) {
              // Update the tracked face with the new data

            /*int trackingID = detectedFace.getTrackingId();
            // Check if the tracking ID is not in the set
            if (!uniqueTrackingIDs.contains(trackingID)) {
              // Add the tracking ID to the set
              uniqueTrackingIDs.add(trackingID);
            }*/

              updatedTrackedFaces.add(detectedFace);
              isSimilar = true;
              break;
            }
          }

          if (!isSimilar) {
            // This is a new face, add it to updatedTrackedFaces
            updatedTrackedFaces.add(detectedFace);
          }
        }

      }

      // Reset the timer each time faces are detected
      //handler.removeCallbacks(timerRunnable);
      //handler.postDelayed(timerRunnable, 5000); // 5 seconds

      liveFaceCount = updatedTrackedFaces.size();
      //liveFaceCount = uniqueTrackingIDs.size();

      trackedFaces = updatedTrackedFaces;

    }catch (Exception e){
      e.printStackTrace();
    }


  }


  private float calculateSimilarityWithMotion(Face detectedFace, Face trackedFace) {
    // Implement your similarity calculation, considering appearance and motion
    // This can involve comparing face landmarks, bounding boxes, and motion information
    // Return a similarity score
    return 0.0f; // Replace with your similarity calculation
  }


  private void updateFaceCountUI() {
    runOnUiThread(() -> faceCountTextView.setText("Live Face Count: " + liveFaceCount+", Total Count: "+totalUniqueFaceCount));
    //Log.e(" updateFaceCountUI() " ,""+ faceCount);
  }

  @Override
  public synchronized void onDestroy() {
    super.onDestroy();
    //sendDemographicStatsToServer(android_id, );
  }

  private void sendDemographicStatsToServer( String data ){

    try {
      errorCode = 400;

      JSONObject jsonObj = new JSONObject(data);
      Gson gson = new Gson();
      Log.e("Data Sent to Server", gson.toJson(jsonObj));


      AndroidNetworking.post("https://xz.sparsatv.in/test/mobileapi/demographicStats.php")
              .addJSONObjectBody(jsonObj)
              .setPriority(Priority.MEDIUM)
              .build()
              .getAsJSONObject(new JSONObjectRequestListener() {
                @Override
                public void onResponse(JSONObject response) {

                  Log.e("RESPONSE", response.toString());

                  try {
                    int status = (int) response.get("status");
                    Log.e("status", ""+status);
                    if (status == 200){
                      Toast.makeText(DetectorActivity.this, "Demographics sent successfully", Toast.LENGTH_SHORT).show();
                      //jsonData
                      dataIntervalList.clear();
                      errorCode = 200;
                      //position++;
                    }else{
                      position++;
                    }
                  } catch (JSONException e) {
                    e.printStackTrace();
                  }

                }

                @Override
                public void onError(ANError error) {
                  // handle error
                  Log.e("Error", error.getErrorDetail());
                  Toast.makeText(DetectorActivity.this, "Demographics was not sent", Toast.LENGTH_SHORT).show();

                }
              });

    } catch (Exception e) {
      e.printStackTrace();
      //Log.e("JSONException:", e.toString());
    }

    /*if(!trackedFaces.isEmpty()){
      handler.removeCallbacks(timerRunnable);
      handler.postDelayed(timerRunnable, 15000);
    }*/

    //Toast.makeText(this, ""+errorCode, Toast.LENGTH_SHORT).show();

  }

  private void updateResults(long currTimestamp, final List<Detector.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currTimestamp);
    trackingOverlay.postInvalidate();
    computingDetection = false;


    runOnUiThread(
            () -> {
              showFrameInfo(previewWidth + "x" + previewHeight);
              showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
              showInference(lastProcessingTimeMs + "ms");
            });

  }

  @Override
  public synchronized void onPause() {
    super.onPause();
    handler.removeCallbacks(runnable);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
