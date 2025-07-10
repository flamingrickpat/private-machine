package com.yourcompany.streamer;

public class NativeBridge {
    static {
        System.loadLibrary("streamer");
    }
    public native long   nativeCreate();
    public native void   nativeStart(long handle, String ip, int port);
    public native void   nativeStop(long handle);
    public native android.graphics.Bitmap nativeSnapshot(long handle, int eye);
}
