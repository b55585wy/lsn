package com.example.biosleepx.ml

import android.content.Context
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class SleepPredictor(context: Context) {
    private var module: Module? = null
    
    init {
        // 从assets加载模型
        val modelFile = File(context.filesDir, "biosleepx_mobile.pt")
        if (!modelFile.exists()) {
            context.assets.open("biosleepx_mobile.pt").use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        module = Module.load(modelFile.absolutePath)
    }
    
    fun predict(eegData: FloatArray): Int {
        // 准备输入数据
        val inputTensor = Tensor.fromBlob(
            eegData,
            longArrayOf(1, 1, eegData.size.toLong())  // batch_size=1, channels=1, length=3000
        )
        
        // 创建EOG数据（全零）
        val eogData = FloatArray(eegData.size) { 0f }
        val eogTensor = Tensor.fromBlob(
            eogData,
            longArrayOf(1, 1, eogData.size.toLong())
        )
        
        // 运行模型
        val output = module?.forward(
            IValue.from(inputTensor),
            IValue.from(eogTensor)
        )?.toTensor()
        
        // 获取预测结果
        val scores = output?.dataAsFloatArray ?: FloatArray(5) { 0f }
        return scores.indices.maxByOrNull { scores[it] } ?: 0
    }
    
    fun close() {
        module?.destroy()
        module = null
    }
} 