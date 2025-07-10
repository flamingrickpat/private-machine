// native_streamer.h
#pragma once
#include <jni.h>
#include <string>
#include <vector>

class PassthroughStreamer {
public:
    static PassthroughStreamer* create(JavaVM* vm, jobject activity);
    void start(const char* ip, uint16_t port,
               int width = 1280, int height = 1024, int fps = 30, int bitrate = 8'000'000);
    void stop();
    jobject snapshot(JNIEnv* env, int eye /*0=L,1=R*/);      // returns Bitmap
private:
    PassthroughStreamer(JavaVM* vm, jobject activity);
    bool openCameras();
    bool setupEncoder(int w,int h,int fps,int br);
    void encodeLoop();
    // …
};
extern "C" {
JNIEXPORT jlong JNICALL
Java_com_yourcompany_streamer_NativeBridge_nativeCreate(JNIEnv*, jobject);
JNIEXPORT void JNICALL
Java_com_yourcompany_streamer_NativeBridge_nativeStart(JNIEnv*, jobject,
jlong handle, jstring ip, jint port);
JNIEXPORT void JNICALL
Java_com_yourcompany_streamer_NativeBridge_nativeStop(JNIEnv*, jobject, jlong handle);
JNIEXPORT jobject JNICALL
        Java_com_yourcompany_streamer_NativeBridge_nativeSnapshot(JNIEnv*, jobject,
                                                                  jlong handle, jint eye);
}
