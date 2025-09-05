package com.example.activitytracker.prefs

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "activity_settings")

data class ThresholdPrefs(
    val sedentaryMax: Double = 0.05,
    val smallMax: Double = 0.12,
    val mediumMax: Double = 0.25
)

class SettingsStore(private val context: Context) {
    private object Keys {
        val SHOW_IMU = booleanPreferencesKey("show_imu")
        val SED_MAX = doublePreferencesKey("sedentary_max")
        val SMALL_MAX = doublePreferencesKey("small_max")
        val MED_MAX = doublePreferencesKey("medium_max")
        val WINDOW_SEC = doublePreferencesKey("window_seconds")
        val STRIDE_SEC = doublePreferencesKey("stride_seconds")
        val THEME_DARK = booleanPreferencesKey("theme_dark")
        val AGE_YEARS = androidx.datastore.preferences.core.intPreferencesKey("age_years")
        val WEIGHT_KG = doublePreferencesKey("weight_kg")
        val HEIGHT_CM = doublePreferencesKey("height_cm")
        val GENDER = androidx.datastore.preferences.core.stringPreferencesKey("gender")
    }

    val showImuFlow: Flow<Boolean> = context.dataStore.data.map { it[Keys.SHOW_IMU] ?: false }
    val thresholdsFlow: Flow<ThresholdPrefs> = context.dataStore.data.map { prefs ->
        ThresholdPrefs(
            sedentaryMax = prefs[Keys.SED_MAX] ?: 0.05,
            smallMax = prefs[Keys.SMALL_MAX] ?: 0.12,
            mediumMax = prefs[Keys.MED_MAX] ?: 0.25
        )
    }

    val windowSecondsFlow: Flow<Double> = context.dataStore.data.map { it[Keys.WINDOW_SEC] ?: 5.0 }
    val strideSecondsFlow: Flow<Double> = context.dataStore.data.map { it[Keys.STRIDE_SEC] ?: 1.0 }

    val themeDarkFlow: Flow<Boolean> = context.dataStore.data.map { it[Keys.THEME_DARK] ?: false }

    val ageYearsFlow: Flow<Int> = context.dataStore.data.map { it[Keys.AGE_YEARS] ?: 0 }
    val weightKgFlow: Flow<Double> = context.dataStore.data.map { it[Keys.WEIGHT_KG] ?: 0.0 }
    val heightCmFlow: Flow<Double> = context.dataStore.data.map { it[Keys.HEIGHT_CM] ?: 0.0 }
    // Restrict to two values; default to "male"
    val genderFlow: Flow<String> = context.dataStore.data.map { it[Keys.GENDER] ?: "male" }

    suspend fun setShowImu(show: Boolean) {
        context.dataStore.edit { it[Keys.SHOW_IMU] = show }
    }

    suspend fun setThresholds(tp: ThresholdPrefs) {
        context.dataStore.edit {
            it[Keys.SED_MAX] = tp.sedentaryMax
            it[Keys.SMALL_MAX] = tp.smallMax
            it[Keys.MED_MAX] = tp.mediumMax
        }
    }

    suspend fun setWindowSeconds(seconds: Double) {
        context.dataStore.edit { it[Keys.WINDOW_SEC] = seconds }
    }

    suspend fun setStrideSeconds(seconds: Double) {
        context.dataStore.edit { it[Keys.STRIDE_SEC] = seconds }
    }

    suspend fun setThemeDark(dark: Boolean) {
        context.dataStore.edit { it[Keys.THEME_DARK] = dark }
    }

    suspend fun setUserProfile(ageYears: Int, weightKg: Double, heightCm: Double, gender: String) {
        context.dataStore.edit {
            it[Keys.AGE_YEARS] = ageYears
            it[Keys.WEIGHT_KG] = weightKg
            it[Keys.HEIGHT_CM] = heightCm
            it[Keys.GENDER] = gender.lowercase()
        }
    }
}
