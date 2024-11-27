package com.example.audiva

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class AnalysisActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        var score = intent.getFloatExtra("score", 0.0f)

        var result = "None"
        var recommendation = "None"
        if (score < 0.5) {
            result = "Result: AI Generated"
            recommendation = "Recommendation: Be careful while interacting with the source of this audio"
        } else {
            result = "Result: Human"
            recommendation = "Recommendation: The source of this audio seems genuine"
        }
        score *= 100
        val scoreText = "Confidence Score: %.2f".format(score)

        val resultTextView: TextView = findViewById(R.id.analysisResult)
        resultTextView.text = result

        val scoreTextView: TextView = findViewById(R.id.analysisConfidence)
        scoreTextView.text = scoreText

        val recomTextView: TextView = findViewById(R.id.analysisRecommendation)
        recomTextView.text = recommendation
    }
}