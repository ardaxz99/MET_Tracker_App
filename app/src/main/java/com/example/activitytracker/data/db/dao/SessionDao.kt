package com.example.activitytracker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import com.example.activitytracker.data.db.entities.ActivitySessionEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface SessionDao {
    @Insert
    suspend fun insert(session: ActivitySessionEntity): Long

    @Update
    suspend fun update(session: ActivitySessionEntity)

    @Query("SELECT * FROM activity_sessions ORDER BY startedAt DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<ActivitySessionEntity>>

    @Query("DELETE FROM activity_sessions")
    suspend fun clearAll()
}
