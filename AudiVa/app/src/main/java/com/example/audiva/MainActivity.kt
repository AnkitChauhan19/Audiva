package com.example.audiva

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.telephony.PhoneStateListener
import android.telephony.TelephonyCallback
import android.telephony.TelephonyManager
import android.telephony.TelephonyCallback.CallStateListener
import android.util.Log
import android.widget.Button
import android.widget.ListView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.json.JSONObject
import java.io.InputStreamReader
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.ceil
import com.jlibrosa.audio.JLibrosa

class MainActivity : AppCompatActivity() {

    private lateinit var telephonyManager: TelephonyManager
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var isRealTimeAnalysisEnabled = false // Flag for real-time analysis toggle
    private val REQUEST_PERMISSION_CODE = 101
    private val REQUEST_NOTIFICATION_PERMISSION_CODE = 102

    // File chooser launcher for selecting audio files
    private lateinit var fileChooserLauncher: ActivityResultLauncher<Intent>

    // TensorFlow Lite model
    private lateinit var tfliteInterpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Load TensorFlow Lite model
        fun loadModelFile(assetManager: AssetManager, fileName: String): ByteBuffer {
            val fileDescriptor = assetManager.openFd(fileName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
//        val modelBuffer = loadModelFile(this.assets, "model.tflite")

//        tfliteInterpreter = Interpreter(modelBuffer, flexDelegate)
        val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
        val flexDelegate = FlexDelegate()
        val options = Interpreter.Options().apply {
            addDelegate(flexDelegate)
        }
        tfliteInterpreter = Interpreter(modelFile, options)

        val realtimeDetectionSwitch = findViewById<SwitchCompat>(R.id.realtimeDetectionSwitch)
        val recentAnalysisList = findViewById<ListView>(R.id.recentAnalysisList)

        // Load the saved state of the toggle switch
        val sharedPref = getSharedPreferences("AppPreferences", MODE_PRIVATE)
        val isSwitchOn = sharedPref.getBoolean("isRealTimeAnalysisEnabled", false) // Default is false if not set
        realtimeDetectionSwitch.isChecked = isSwitchOn
        isRealTimeAnalysisEnabled = isSwitchOn // Update the flag as well

        // Initialize file chooser
        fileChooserLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val audioUri = result.data?.data
                if (audioUri != null) {
                    handleAudioFile(audioUri)
                }
            }
        }

        val uploadAudioButton = findViewById<Button>(R.id.uploadAudioButton)
        uploadAudioButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                type = "audio/*"
                addCategory(Intent.CATEGORY_OPENABLE)
            }
            fileChooserLauncher.launch(Intent.createChooser(intent, "Select Audio"))
        }

        // Real-time Detection Toggle Switch Listener
        realtimeDetectionSwitch.setOnCheckedChangeListener { _, isChecked ->
            isRealTimeAnalysisEnabled = isChecked
            // Save the state in SharedPreferences
            with (sharedPref.edit()) {
                putBoolean("isRealTimeAnalysisEnabled", isChecked)
                apply()
            }
        }

        // Request permissions
        if (!hasPermissions()) {
            requestPermissions()
        }

        // Load recent analysis data
        loadRecentAnalysis(recentAnalysisList)

        // Set up call callback for call detection
        setupCallDetection()

        // Set up item click listener for recentAnalysisList
        recentAnalysisList.setOnItemClickListener { _, _, position, _ ->
            val intent = Intent(this, AnalysisActivity::class.java)
            intent.putExtra("EXTRA_ITEM_POSITION", position) // Optionally pass the item position or other data
            startActivity(intent)
        }

    }

    private fun loadScalerParams(context: Context): Pair<FloatArray, FloatArray> {
        val inputStream = context.assets.open("scaler_params.json")
        val reader = InputStreamReader(inputStream)
        val json = JSONObject(reader.readText())

        val meanArray = json.getJSONArray("mean")
        val scaleArray = json.getJSONArray("scale")

        val mean = FloatArray(meanArray.length()) { meanArray.getDouble(it).toFloat() }
        val scale = FloatArray(scaleArray.length()) { scaleArray.getDouble(it).toFloat() }

        return Pair(mean, scale)
    }

    private fun scaleMFCC(mfcc: FloatArray, mean: FloatArray, scale: FloatArray): FloatArray {
        // Apply the scaling: (value - mean) / scale
        return mfcc.mapIndexed { index, value ->
            (value - mean[index]) / scale[index]
        }.toFloatArray()
    }

    private fun handleAudioFile(uri: Uri) {
        // Process the uploaded audio file
        Toast.makeText(this, "Audio file selected: $uri", Toast.LENGTH_SHORT).show()
        val audioFile = contentResolver.openInputStream(uri)
        audioFile?.let { processUploadedAudioFile(it) }
    }

    private fun processUploadedAudioFile(inputStream: InputStream) {
        val sampleRate = 22050
        val context = applicationContext
        val mfccFeatures = preprocessAudio(inputStream, sampleRate, context)

        val (mean, scale) = loadScalerParams(context)

        val predictions = mutableListOf<Float>()
        for (i in 0 until mfccFeatures.size) {
            val curMfcc = mfccFeatures[i]
            Log.d("MFCC_Values", "MFCCs: ${curMfcc.joinToString(", ")}")
            val scaledMfccFeatures = scaleMFCC(curMfcc, mean, scale)
            Log.d("Scaled_MFCC", "Scaled: ${scaledMfccFeatures.joinToString(", ")}")
            val curPrediction = analyzeUploadedWithCustomModel(scaledMfccFeatures)
            if (curPrediction > 0 && curPrediction < 1) {
                predictions.add(curPrediction)
            }
        }

        var aiScore = 0F
        for (i in 0 until predictions.size) {
            aiScore += predictions[i]
        }

        aiScore /= predictions.size
        val intent = Intent(this, AnalysisActivity::class.java)
        intent.putExtra("score", aiScore)
        startActivity(intent)
    }

    private fun analyzeUploadedWithCustomModel(audioBuffer: FloatArray): Float {
        // Run the model inference on the preprocessed audio data
        val reshapedInput = FloatArray(13 * 1) { 0f }
        for (i in audioBuffer.indices) {
            reshapedInput[i] = audioBuffer[i] // Assuming audioBuffer is scaled correctly
        }

        val input = ByteBuffer.allocateDirect(4 * audioBuffer.size)
        input.order(ByteOrder.nativeOrder())
        input.asFloatBuffer().put(audioBuffer)

        // Allocate output tensor
        val output = Array(1) { FloatArray(1) }
        tfliteInterpreter.run(input, output)

        // Return AI classification result (this is just an example; adjust according to model output)
        return output[0][0]  // Example threshold; update based on your model's output
    }

    private fun preprocessAudio(inputStream: InputStream, sampleRate: Int, context: Context): MutableList<FloatArray> {
        // Convert InputStream to PCM data (ShortArray)
        val audioBuffer = convertInputStreamToPCM(inputStream, sampleRate)

        val twoSecondAudioLength = sampleRate * 2  // 2 seconds worth of samples
        val numChunks = audioBuffer.size / twoSecondAudioLength  // Number of 2-second chunks

        // Prepare an array to hold the MFCC features for each chunk
        val mfccFeaturesList = mutableListOf<FloatArray>()

        // Process each 2-second chunk of audio
        for (i in 0 until numChunks) {
            val startIdx = i * twoSecondAudioLength
            val endIdx = startIdx + twoSecondAudioLength

            // Slice the audio buffer for this chunk
            val chunk = audioBuffer.copyOfRange(startIdx, endIdx)

            // Compute MFCC features for the chunk
            val mfccFeatures = extractMfcc(chunk, sampleRate)

            // Add the MFCC features to the list
            mfccFeaturesList.add(mfccFeatures)
        }

        // Calculate the mean of the MFCC features across all chunks
//        val meanMfcc = calculateMeanMfcc(mfccFeaturesList)

        return mfccFeaturesList
    }

    // Function to convert the InputStream into PCM data (ShortArray)
    private fun convertInputStreamToPCM(inputStream: InputStream, sampleRate: Int): ShortArray {
        val byteArrayOutputStream = ByteArrayOutputStream()

        // Read the InputStream into a byte array
        val buffer = ByteArray(2048)
        var bytesRead: Int
        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
            byteArrayOutputStream.write(buffer, 0, bytesRead)
        }

        // Convert byte array to PCM data (ShortArray)
        val byteArray = byteArrayOutputStream.toByteArray()
        val shortArray = ShortArray(byteArray.size / 2)

        for (i in shortArray.indices) {
            shortArray[i] = (byteArray[2 * i].toInt() or (byteArray[2 * i + 1].toInt() shl 8)).toShort()
        }

        return shortArray
    }

    // Function to extract MFCC features from an audio chunk
    private fun extractMfcc(audioChunk: ShortArray, sampleRate: Int): FloatArray {
        // Convert ShortArray to FloatArray for MFCC processing
        val audioFloatArray = audioChunk.map { it.toFloat() / Short.MAX_VALUE }.toFloatArray()

        val mfcc = JLibrosa()
        mfcc.sampleRate = sampleRate


        val mfccs = mfcc.generateMFCCFeatures(audioFloatArray, sampleRate, 13)
        return mfcc.generateMeanMFCCFeatures(mfccs, 13, mfccs.size)
    }

    private fun calculateMeanMfcc(mfccList: List<FloatArray>): FloatArray {
        if (mfccList.isEmpty()) {
            return FloatArray(0) // or return some default value
        }

        val numFeatures = mfccList[0].size
        val meanMfcc = FloatArray(numFeatures)

        // Initialize the mean MFCC array
        for (i in meanMfcc.indices) {
            var sum = 0.0f
            for (mfcc in mfccList) {
                sum += mfcc[i]
            }
            meanMfcc[i] = sum / mfccList.size
        }

        return meanMfcc
    }

    private fun hasPermissions(): Boolean {
        return (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED)
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.READ_PHONE_STATE),
            REQUEST_PERMISSION_CODE
        )
    }

    private fun loadRecentAnalysis(recentAnalysisList: ListView) {
        // Sample data
        val analysisData = listOf(
            PhoneNumberAnalysis("123-456-7890", true),  // AI-voiced - caution icon
            PhoneNumberAnalysis("098-765-4321", false)  // Not AI-voiced - check icon
        )

        // Set custom adapter
        val adapter = AnalysisAdapter(this, analysisData)
        recentAnalysisList.adapter = adapter
    }

    private fun setupCallDetection() {
        // Ensure permissions are granted before accessing telephony features
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED) {
            telephonyManager = getSystemService(TELEPHONY_SERVICE) as TelephonyManager

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                // Use TelephonyCallback for Android 12 (API level 31) and above
                telephonyManager.registerTelephonyCallback(mainExecutor, object : TelephonyCallback(), CallStateListener {
                    // Correct implementation of TelephonyCallback
                    override fun onCallStateChanged(state: Int) {
                        if (isRealTimeAnalysisEnabled) {
                            when (state) {
                                TelephonyManager.CALL_STATE_OFFHOOK -> startRealTimeAudioCapture() // Call picked up
                                TelephonyManager.CALL_STATE_IDLE -> stopRealTimeAudioCapture() // Call ended
                            }
                        }
                    }
                })
            } else {
                // Use PhoneStateListener for older versions
                @Suppress("DEPRECATION")
                telephonyManager.listen(phoneStateListener, PhoneStateListener.LISTEN_CALL_STATE)
            }
        } else {
            // Request permission if not granted
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_PHONE_STATE),
                REQUEST_PERMISSION_CODE
            )
        }
    }

    @Suppress("DEPRECATION")
    private val phoneStateListener = object : PhoneStateListener() {
        @Deprecated("Deprecated in Java")
        override fun onCallStateChanged(state: Int, phoneNumber: String?) {
            if (isRealTimeAnalysisEnabled) {
                when (state) {
                    TelephonyManager.CALL_STATE_OFFHOOK -> startRealTimeAudioCapture() // Call picked up
                    TelephonyManager.CALL_STATE_IDLE -> stopRealTimeAudioCapture() // Call ended
                    TelephonyManager.CALL_STATE_RINGING -> {
                        TODO()
                    }
                }
            }
        }
    }

    private fun startRealTimeAudioCapture() {
        // Check for audio record permission before starting audio capture
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            val sampleRate = 22050
//            val bufferSize = AudioRecord.getMinBufferSize(
//                sampleRate,
//                AudioFormat.CHANNEL_IN_MONO,
//                AudioFormat.ENCODING_PCM_16BIT
//            )
            val bufferSize = sampleRate * 4

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            audioRecord?.startRecording()
            isRecording = true

            processAudioData(bufferSize)
        } else {
            // Request permission if not granted
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_PERMISSION_CODE
            )
        }
    }

    private fun processAudioData(bufferSize: Int) {
        CoroutineScope(Dispatchers.IO).launch {
            val audioBuffer = ShortArray(bufferSize)

            while (isRecording) {
                val readSize = audioRecord?.read(audioBuffer, 0, bufferSize) ?: 0
                if (readSize > 0) {
                    val sampleRate = 22050
                    val context = applicationContext
                    val buffer = preprocessAudio(audioBuffer, sampleRate, context)

                    val (mean, scale) = loadScalerParams(context)

                    val predictions = mutableListOf<Float>()
                    for (i in 0 until buffer.size) {
                        val curMfcc = buffer[i]
                        val scaledMfccFeatures = scaleMFCC(curMfcc, mean, scale)
                        val curPrediction = analyzeUploadedWithCustomModel(scaledMfccFeatures)
                        if (curPrediction > 0 && curPrediction < 1) {
                            predictions.add(curPrediction)
                        }
                    }

                    var aiScore = 0F
                    for (i in 0 until predictions.size) {
                        aiScore += predictions[i]
                    }

                    aiScore /= predictions.size

                    if (aiScore < 0.5) {
                        sendDetectionNotification()
                    }
                    else {
                        sendHumanNotification()
                    }
                }
            }
        }
    }
    private fun preprocessAudio(audioBuffer: ShortArray, sampleRate: Int, context: Context): MutableList<FloatArray> {
        val twoSecondAudioLength = sampleRate * 2  // 2 seconds worth of samples
        val numChunks = audioBuffer.size / twoSecondAudioLength  // Number of 2-second chunks

        // Prepare an array to hold the MFCC features for each chunk
        val mfccFeaturesList = mutableListOf<FloatArray>()

        // Process each 2-second chunk of audio
        for (i in 0 until numChunks) {
            val startIdx = i * twoSecondAudioLength
            val endIdx = startIdx + twoSecondAudioLength

            // Slice the audio buffer for this chunk
            val chunk = audioBuffer.copyOfRange(startIdx, endIdx)

            // Compute MFCC features for the chunk
            val mfccFeatures = extractMfcc(chunk, sampleRate)

            // Add the MFCC features to the list
            mfccFeaturesList.add(mfccFeatures)
        }

        // Calculate the mean of the MFCC features across all chunks
//        val meanMfcc = calculateMeanMfcc(mfccFeaturesList)

        return mfccFeaturesList
    }

    private fun stopRealTimeAudioCapture() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun sendDetectionNotification() {
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "ai_voice_detection"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "AI Voice Detection", NotificationManager.IMPORTANCE_HIGH)
            notificationManager.createNotificationChannel(channel)
        }

        // For Android 13+ (API level 33), ensure you have the POST_NOTIFICATIONS permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATION_PERMISSION_CODE)
            }
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("AI Voice Detected")
            .setContentText("AI voice detected during ongoing call.")
            .setSmallIcon(R.drawable.caution_7805444) // Replace with your icon
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()

        notificationManager.notify(1, notification)
    }

    private fun sendHumanNotification() {
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "ai_voice_detection"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "AI Voice Detection", NotificationManager.IMPORTANCE_HIGH)
            notificationManager.createNotificationChannel(channel)
        }

        // For Android 13+ (API level 33), ensure you have the POST_NOTIFICATIONS permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATION_PERMISSION_CODE)
            }
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Human Voice Detected")
            .setContentText("Human voice detected during ongoing call.")
            .setSmallIcon(R.drawable.check_5610944) // Replace with your icon
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()

        notificationManager.notify(1, notification)
    }

    // Handle permission result
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_PERMISSION_CODE -> {
                if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                    // Permissions granted, proceed with functionality
                } else {
                    Toast.makeText(this, "Permissions denied", Toast.LENGTH_SHORT).show()
                }
            }
            REQUEST_NOTIFICATION_PERMISSION_CODE -> {
                if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                    // Permissions granted, show notifications
                } else {
                    Toast.makeText(this, "Notification permission denied", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
}
