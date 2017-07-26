package com.stu.wujian.qbyedemo.util;

/**
 * Created by wujian on 17-7-22.
 */

public class EnergyVad {

    private enum Status {KActive, KSilence};
    private float energyThres;
    private int windowSize;
    private int step;
    private int segmentNum;
    private Status currentStatus;

    public EnergyVad(float threshold, int window) {
        energyThres = threshold;
        windowSize = window;
        currentStatus = Status.KSilence;
        step = segmentNum = 0;
    }

    public int getSegmentNum() {
        return segmentNum;
    }

    public boolean isSilence() {
        return currentStatus == Status.KSilence;
    }

    public void run(float energy) {
        boolean succ = energy > energyThres;
        switch (currentStatus) {
            case KSilence:
                if (succ) {
                    step++;
                    if (step == windowSize) {
                        currentStatus = Status.KActive;
                    }
                } else {
                    step = 0;
                }
                break;
            case KActive:
                if (succ) {
                    step = windowSize;
                } else {
                    step--;
                    if (step == 0) {
                        currentStatus = Status.KSilence;
                        segmentNum++;
                    }
                }
                break;
            default:
                break;
        }
    }
}
