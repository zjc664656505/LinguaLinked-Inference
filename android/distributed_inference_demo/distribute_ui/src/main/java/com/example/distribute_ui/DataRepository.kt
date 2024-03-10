package com.example.distribute_ui
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

object DataRepository {
    private val _isDirEmptyLiveData = MutableLiveData<Boolean>()
    val isDirEmptyLiveData: LiveData<Boolean> = _isDirEmptyLiveData

    private val _decodingStringLiveData = MutableLiveData<String>()
    val decodingStringLiveData: LiveData<String> = _decodingStringLiveData

    private val _sampleId = MutableLiveData<Int>()
    val sampleId: LiveData<Int> = _sampleId
    fun updateSampleId(sampleId: Int) {
        _sampleId.postValue(sampleId)
    }

    fun updateDecodingString(updatedString: String) {
//        val responsePosition: Int = updatedString.indexOf("Response:")
//        val decodedStringAfterResponse: String = updatedString.substring(responsePosition + 9)
        _decodingStringLiveData.postValue(updatedString)
    }

    fun setIsDirEmpty(isEmpty: Boolean) {
        _isDirEmptyLiveData.postValue(isEmpty)
    }
}