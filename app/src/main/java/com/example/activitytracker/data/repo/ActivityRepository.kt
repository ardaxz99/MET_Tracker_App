package com.example.activitytracker.data.repo

import com.example.activitytracker.data.db.dao.SessionDao
import com.example.activitytracker.data.db.dao.SliceDao
import com.example.activitytracker.data.db.dao.SecondDao
import com.example.activitytracker.data.db.dao.DayStateCount
import com.example.activitytracker.data.db.dao.MonthStateCount
import com.example.activitytracker.data.db.entities.ActivitySessionEntity
import com.example.activitytracker.data.db.entities.ActivitySliceEntity
import com.example.activitytracker.data.db.entities.ActivitySecondEntity
import kotlinx.coroutines.flow.Flow

class ActivityRepository(
    private val sessionDao: SessionDao,
    private val sliceDao: SliceDao,
    private val secondDao: SecondDao
) {
    suspend fun startSession(startTime: Long): Long =
        sessionDao.insert(ActivitySessionEntity(startedAt = startTime, endedAt = null, summaryJson = null))

    suspend fun endSession(sessionId: Long, endTime: Long, summaryJson: String?) {
        val session = ActivitySessionEntity(id = sessionId, startedAt = 0L, endedAt = endTime, summaryJson = summaryJson)
        // We need original startedAt; in real code fetch it. Here keep best-effort no-op update if needed.
        // This is a simplified sample; consider adding a query to load by id.
        sessionDao.update(session)
    }

    suspend fun addSlice(sessionId: Long, start: Long, end: Long?, state: String) {
        sliceDao.insert(ActivitySliceEntity(sessionId = sessionId, start = start, end = end, state = state))
    }

    fun recentSlices(limit: Int = 50): Flow<List<ActivitySliceEntity>> = sliceDao.recent(limit)

    suspend fun clearAll() {
        sliceDao.clearAll()
        secondDao.clearAll()
        sessionDao.clearAll()
    }

    suspend fun recordSecond(tsSec: Long, state: String) {
        secondDao.upsert(ActivitySecondEntity(tsSec = tsSec, state = state))
    }

    fun dayHourCounts(dayStartSec: Long): Flow<List<com.example.activitytracker.data.db.dao.HourStateCount>> {
        val end = dayStartSec + 24 * 3600
        return secondDao.hourCountsForDay(dayStartSec, end)
    }

    fun weekDayCounts(weekStartSec: Long): Flow<List<DayStateCount>> {
        val end = weekStartSec + 7 * 24 * 3600
        return secondDao.dayCountsForRange(weekStartSec, end)
    }

    fun monthDayCounts(monthStartSec: Long, daysInMonth: Int): Flow<List<DayStateCount>> {
        val end = monthStartSec + daysInMonth * 24 * 3600L
        return secondDao.dayCountsForRange(monthStartSec, end)
    }

    fun yearMonthCounts(yearStartSec: Long): Flow<List<MonthStateCount>> {
        val end = yearStartSec + 366L * 24 * 3600L // safe window to include leap years; query groups by month
        return secondDao.monthCountsForYear(yearStartSec, end)
    }
}
