package org.tensorflow.lite.examples.detection;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;

public class DemographicsData {

    @SerializedName("hardwareKey")
    @Expose
    private String hardware_key;

    @SerializedName("data")
    @Expose
    private List<DataInterval> data;

    public String getHardwareKey() {
        return hardware_key;
    }

    public void setHardwareKey(String hardware_key) {
        this.hardware_key = hardware_key;
    }

    public List<DataInterval> getData() {
        return data;
    }

    public void setData(List<DataInterval> data) {
        this.data = data;
    }

    public static class DataInterval {
        @SerializedName("peopleCount")
        @Expose
        private int people_count;
        @SerializedName("startTime")
        @Expose
        private String start_time;
        @SerializedName("endTime")
        @Expose
        private String end_time;
        /*@SerializedName("trackingId")
        @Expose
        private int trackingId;*/


        public int getPeopleCount() {
            return people_count;
        }

        public void setPeopleCount(int people_count) {
            this.people_count = people_count;
        }

        public String getStartTime() {
            return start_time;
        }

        public void setStartTime(String start_time) {
            this.start_time = start_time;
        }

        public String getEndTime() {
            return end_time;
        }

        public void setEndTime(String end_time) {
            this.end_time = end_time;
        }
        /*public int getTrackingId() {
            return trackingId;
        }
        public void setTrackingId(int trackingId) {
            this.trackingId = trackingId;
        }*/
    }

}
