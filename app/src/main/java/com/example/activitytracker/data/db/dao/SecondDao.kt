package com.example.activitytracker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

data class HourStateCount(
    val hour: Int,
    val state: String,
    val cnt: Int
)

data class DayStateCount(
    val dayIndex: Int,
    val state: String,
    val cnt: Int
)

data class MonthStateCount(
    val monthIndex: Int,
    val state: String,
    val cnt: Int
)

@Dao
interface SecondDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(ts: com.example.activitytracker.data.db.entities.ActivitySecondEntity)

    @Query("DELETE FROM activity_seconds")
    suspend fun clearAll()

    // Counts per hour for a given day window [startSec, endSec)
    @Query(
        "SELECT CAST(strftime('%H', datetime(tsSec, 'unixepoch')) AS INTEGER) AS hour, state, COUNT(*) AS cnt " +
        "FROM activity_seconds WHERE tsSec >= :startSec AND tsSec < :endSec " +
        "GROUP BY hour, state ORDER BY hour ASC"
    )
    fun hourCountsForDay(startSec: Long, endSec: Long): kotlinx.coroutines.flow.Flow<List<HourStateCount>>

    // Counts per day index for an arbitrary range [startSec, endSec), dayIndex starts at 0
    @Query(
        "SELECT CAST((tsSec - :startSec)/86400 AS INTEGER) AS dayIndex, state, COUNT(*) AS cnt " +
        "FROM activity_seconds WHERE tsSec >= :startSec AND tsSec < :endSec " +
        "GROUP BY dayIndex, state ORDER BY dayIndex ASC"
    )
    fun dayCountsForRange(startSec: Long, endSec: Long): kotlinx.coroutines.flow.Flow<List<DayStateCount>>

    // Counts per month (0..11) within a calendar year range [startSec, endSec)
    @Query(
        "SELECT CAST(strftime('%m', datetime(tsSec, 'unixepoch')) AS INTEGER) - 1 AS monthIndex, state, COUNT(*) AS cnt " +
        "FROM activity_seconds WHERE tsSec >= :startSec AND tsSec < :endSec " +
        "GROUP BY monthIndex, state ORDER BY monthIndex ASC"
    )
    fun monthCountsForYear(startSec: Long, endSec: Long): kotlinx.coroutines.flow.Flow<List<MonthStateCount>>
}
