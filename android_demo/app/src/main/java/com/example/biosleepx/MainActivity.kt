package com.example.biosleepx

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.example.biosleepx.data.DemoData
import com.example.biosleepx.data.Metadata
import com.example.biosleepx.ml.SleepPredictor
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {
    private lateinit var eegChart: LineChart
    private lateinit var stageText: TextView
    private lateinit var featuresText: TextView
    private lateinit var predictor: SleepPredictor
    private lateinit var metadata: Metadata
    private lateinit var demoData: Map<String, List<DemoData>>
    private var currentStage = 0
    private var currentSegmentIndex = 0
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // 初始化视图
        eegChart = findViewById(R.id.eegChart)
        stageText = findViewById(R.id.stageText)
        featuresText = findViewById(R.id.featuresText)
        
        // 配置图表
        setupChart()
        
        // 加载模型
        predictor = SleepPredictor(this)
        
        // 加载演示数据
        loadDemoData()
        
        // 设置按钮点击事件
        setupButtons()
        
        // 显示初始数据
        showStageData(0, 0)
    }
    
    private fun setupChart() {
        eegChart.apply {
            description.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)
            
            xAxis.apply {
                setDrawGridLines(false)
                setDrawAxisLine(true)
            }
            
            axisLeft.apply {
                setDrawGridLines(true)
                setDrawAxisLine(true)
            }
            
            axisRight.isEnabled = false
            legend.isEnabled = false
        }
    }
    
    private fun loadDemoData() {
        val gson = Gson()
        
        // 加载元数据
        InputStreamReader(assets.open("metadata.json")).use { reader ->
            metadata = gson.fromJson(reader, Metadata::class.java)
        }
        
        // 加载演示数据
        val type = object : TypeToken<Map<String, List<DemoData>>>() {}.type
        InputStreamReader(assets.open("demo_segments.json")).use { reader ->
            demoData = gson.fromJson(reader, type)
        }
    }
    
    private fun setupButtons() {
        // 设置阶段选择按钮
        val stageButtons = mapOf(
            R.id.btnWake to 0,
            R.id.btnN1 to 1,
            R.id.btnN2 to 2,
            R.id.btnN3 to 3,
            R.id.btnREM to 4
        )
        
        stageButtons.forEach { (buttonId, stageId) ->
            findViewById<Button>(buttonId).setOnClickListener {
                currentStage = stageId
                currentSegmentIndex = 0
                showStageData(currentStage, currentSegmentIndex)
            }
        }
        
        // 设置导航按钮
        findViewById<Button>(R.id.btnPrev).setOnClickListener {
            if (currentSegmentIndex > 0) {
                currentSegmentIndex--
                showStageData(currentStage, currentSegmentIndex)
            }
        }
        
        findViewById<Button>(R.id.btnNext).setOnClickListener {
            val maxIndex = (demoData[currentStage.toString()]?.size ?: 1) - 1
            if (currentSegmentIndex < maxIndex) {
                currentSegmentIndex++
                showStageData(currentStage, currentSegmentIndex)
            }
        }
    }
    
    private fun showStageData(stageId: Int, segmentIndex: Int) {
        // 获取该阶段的数据
        val segments = demoData[stageId.toString()] ?: return
        if (segmentIndex >= segments.size) return
        
        val segment = segments[segmentIndex]
        
        // 更新图表
        val entries = segment.data.mapIndexed { index, value ->
            Entry(index.toFloat(), value)
        }
        
        val dataSet = LineDataSet(entries, "EEG").apply {
            color = getColor(R.color.purple_500)
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 2f
        }
        
        eegChart.data = LineData(dataSet)
        eegChart.invalidate()
        
        // 运行模型预测
        val prediction = predictor.predict(segment.data.toFloatArray())
        
        // 更新UI
        val predictedStage = metadata.stage_mapping[prediction.toString()] ?: "Unknown"
        val actualStage = metadata.stage_mapping[segment.label.toString()] ?: "Unknown"
        stageText.text = "Predicted: $predictedStage\nActual: $actualStage"
        
        // 显示频谱特征
        val features = segment.features
        featuresText.text = """
            Frequency Band Powers:
            Delta (0.5-4 Hz): %.2f
            Theta (4-8 Hz): %.2f
            Alpha (8-13 Hz): %.2f
            Beta (13-30 Hz): %.2f
        """.trimIndent().format(
            features.delta_power,
            features.theta_power,
            features.alpha_power,
            features.beta_power
        )
        
        // 更新导航按钮状态
        findViewById<Button>(R.id.btnPrev).isEnabled = segmentIndex > 0
        findViewById<Button>(R.id.btnNext).isEnabled = segmentIndex < segments.size - 1
    }
    
    override fun onDestroy() {
        super.onDestroy()
        predictor.close()
    }
} 