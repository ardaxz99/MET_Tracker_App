package com.example.activitytracker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import com.example.activitytracker.data.db.entities.ActivitySliceEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface SliceDao {
    @Insert
    suspend fun insert(slice: ActivitySliceEntity): Long

    @Update
    suspend fun update(slice: ActivitySliceEntity)

    @Query("SELECT * FROM activity_slices WHERE sessionId = :sessionId ORDER BY start ASC")
    fun bySession(sessionId: Long): Flow<List<ActivitySliceEntity>>

    @Query("SELECT * FROM activity_slices ORDER BY start DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<ActivitySliceEntity>>

    @Query("DELETE FROM activity_slices")
    suspend fun clearAll()
}

