package com.stu.wujian.qbyedemo.util;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.util.Log;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Created by wujian on 17-7-22.
 */

public class EnrollmentThread extends Thread {

    private AudioRecord audioRecord;
    private Handler uiHandler;
    private Context activityContext;
    private int minBufferSize;

    private final int FREQ = 16000;
    private final int CONF = AudioFormat.CHANNEL_CONFIGURATION_MONO;
    private final int CODE = AudioFormat.ENCODING_PCM_16BIT;

    private final int FRAME_LEN = 400;
    private final int FRAME_OFF = 160;
    private final int LOG_MSG = 2;

    private final int TOAST_ROUND = 4;


    public EnrollmentThread(Handler handler, Context context) {
        uiHandler = handler;
        minBufferSize = AudioRecord.getMinBufferSize(FREQ, CONF, CODE);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, FREQ, CONF, CODE, minBufferSize);
        activityContext = context;
    }

    @Override
    public void run() {
        boolean status = true;

        for (int t = 0; t < 3; t++) {

            short[] readBuffer = new short[minBufferSize];
            short[] remainBuffer = new short[FRAME_LEN];
            int remainSize = 0, frameNum = 0, startFrame = 0;

            audioRecord.startRecording();
            EnergyVad energyVad = new EnergyVad(10, 30);
            boolean stop = false;

            uiHandler.obtainMessage(LOG_MSG, t == 0 ? "Now Say Keywords": "Repeat that...").sendToTarget();
            DataOutputStream dataOutputStream = null;

            try {
                dataOutputStream = new DataOutputStream(activityContext.openFileOutput("templ.pcm", activityContext.MODE_PRIVATE));
            } catch (IOException e) {
                Log.i("KALDI", "EnrollmentThread: init dataOutputStream failed");
                uiHandler.obtainMessage(LOG_MSG, "init dataOutputStream failed").sendToTarget();
                status = false;
            }

            while (true && dataOutputStream != null) {

                int hasread = audioRecord.read(readBuffer, 0, readBuffer.length);

                try {
                    for (int i = 0; i < hasread; i++)
                        dataOutputStream.writeShort(readBuffer[i]);
                } catch (IOException e) {
                    Log.i("KALDI", "EnrollmentThread: dataOutputStream write short failed");
                    status = false;
                    break;
                }

                int nframes = (hasread + remainSize - FRAME_LEN) / FRAME_OFF + 1;

                short[] waveform = new short[hasread + remainSize];
                System.arraycopy(remainBuffer, 0, waveform, 0, remainSize);
                System.arraycopy(readBuffer, 0, waveform, remainSize, hasread);

                for (int i = 0; i < nframes; i++) {

                    double energy = 0.0f;
                    for (int j = i * FRAME_OFF; j < i * FRAME_OFF + FRAME_LEN; j++) {
                        energy += waveform[j] * waveform[j];
                    }
                    energy = Math.sqrt(energy) / FRAME_LEN;

                    /*
                    if (frameNum % TOAST_ROUND == 0) {
                        String msg = "";
                        for (int j = 0; j < energy / 5 + 1; j++)
                            msg += "|";
                        uiHandler.obtainMessage(LOG_MSG, msg).sendToTarget();
                    }
                    */

                    energyVad.run((float) energy);

                    if (!energyVad.isSilence() && startFrame == 0) {
                        startFrame = frameNum;
                    }

                    if (energyVad.getSegmentNum() >= 1) {
                        stop = true;
                        break;
                    }
                    frameNum++;
                }
                if (stop)
                    break;

                // recalculate the remain size
                remainSize = waveform.length - nframes * FRAME_OFF;
                System.arraycopy(waveform, nframes * FRAME_OFF, remainBuffer, 0, remainSize);
            }

            Log.i("KALDI", "[" + startFrame + ", " + frameNum + "]");
            uiHandler.obtainMessage(LOG_MSG, "Processing...").sendToTarget();

            DataInputStream dataInputStream = null;
            try {
                if (dataOutputStream != null)
                    dataOutputStream.close();
                dataInputStream = new DataInputStream(activityContext.openFileInput("templ.pcm"));
                int startPos = (startFrame - 30) * FRAME_OFF, pos = 0;
                int stopPos = (frameNum - 10) * FRAME_OFF + FRAME_LEN;
                short[] template = new short[stopPos - startPos + 1];

                while (dataInputStream.available() > 0) {
                    short data = dataInputStream.readShort();
                    if (pos >= startPos)
                        template[pos - startPos] = data;
                    if (pos >= stopPos)
                        break;
                    pos++;
                }
                KaldiLib kaldiLib = new KaldiLib();

                if (!kaldiLib.computePosts(template))
                    status = false;

            } catch (IOException e) {
                Log.i("KALDI", "EnrollmentThread: write posteriors failed");
                status = false;
            }

            audioRecord.stop();
        }

        uiHandler.obtainMessage(LOG_MSG, "Enrollment " + (status ? "Done": "Failed")).sendToTarget();
    }
}
