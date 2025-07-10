// native_passthrough_streamer.cpp  — REVISED (2025‑07‑09 d)
// -----------------------------------------------------------------------------
// • Fix NDK r27.2 signature: ACameraCaptureSession_setRepeatingRequest(session, callbacks, numRequests, requests, seqId)
//   Added explicit nullptr for callbacks and made requests array non‑const.
// -----------------------------------------------------------------------------

#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>

#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCaptureRequest.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <cstring>
#include <memory>

#ifndef LOG_TAG
#define LOG_TAG "PassthroughStreamer"
#endif
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

constexpr int CAM_ID_MAX = 32;   // generous buffer for camera IDs

// -----------------------------------------------------------------------------
struct PacketHdr { uint8_t eye; uint8_t flags; uint16_t seq; uint32_t timestamp; uint32_t frameId; } __attribute__((packed));

class PassthroughStreamer {
public:
    static PassthroughStreamer* create(JavaVM* vm, jobject act){ return new PassthroughStreamer(vm, act); }

    // -------------------------- Public entry points -------------------------
    void start(const char* ip, uint16_t port,int w=1280,int h=1024,int fps=30,int br=8'000'000){
        if(running_) return;
        running_=true; frameId_[0]=frameId_[1]=0; startTime_=std::chrono::steady_clock::now();
        sock_=socket(AF_INET,SOCK_DGRAM,0); if(sock_<0){LOGE("socket()"); running_=false; return;}
        addr_={}; addr_.sin_family=AF_INET; addr_.sin_port=htons(port); inet_pton(AF_INET,ip,&addr_.sin_addr);
        if(!initCamera(w,h,fps,br,0)||!initCamera(w,h,fps,br,1)){running_=false; return;}
        worker_[0]=std::thread(&PassthroughStreamer::encodeLoop,this,0);
        worker_[1]=std::thread(&PassthroughStreamer::encodeLoop,this,1);
        LOGI("Streamer ➜ %s:%u",ip,port);
    }
    void stop(){ if(!running_) return; running_=false; for(auto& t:worker_) if(t.joinable()) t.join();
        for(int i=0;i<2;++i){ if(codec_[i]){AMediaCodec_stop(codec_[i]);AMediaCodec_delete(codec_[i]);codec_[i]=nullptr;}
            if(session_[i]){ACameraCaptureSession_stopRepeating(session_[i]);ACameraCaptureSession_close(session_[i]);session_[i]=nullptr;}
            if(device_[i]) {ACameraDevice_close(device_[i]); device_[i]=nullptr;} }
        if(sock_>=0){close(sock_); sock_=-1;}
    }
    jobject snapshot(JNIEnv*,int){ return nullptr; }
    ~PassthroughStreamer(){ stop(); if(cameraMgr_) ACameraManager_delete(cameraMgr_); if(activity_) envMain_->DeleteGlobalRef(activity_); }

private:
    PassthroughStreamer(JavaVM* vm,jobject act):vm_(vm){ vm_->GetEnv((void**)&envMain_,JNI_VERSION_1_6); activity_=envMain_->NewGlobalRef(act); cameraMgr_=ACameraManager_create(); }

    // ------------------------------ Camera utils ---------------------------
    bool getIds(char ids[2][CAM_ID_MAX]){
        ACameraIdList* list=nullptr; if(ACameraManager_getCameraIdList(cameraMgr_,&list)!=ACAMERA_OK) return false;
        bool l=false,r=false; const uint32_t TAG_SRC=0xE100, TAG_POS=0xE101;
        for(int i=0;i<list->numCameras;++i){ const char* id=list->cameraIds[i];
            ACameraMetadata* m=nullptr; if(ACameraManager_getCameraCharacteristics(cameraMgr_,id,&m)!=ACAMERA_OK) continue;
            ACameraMetadata_const_entry eSrc,ePos; bool ok=ACameraMetadata_getConstEntry(m,TAG_SRC,&eSrc)==ACAMERA_OK && eSrc.data.u8[0]==0 &&
                                                           ACameraMetadata_getConstEntry(m,TAG_POS,&ePos)==ACAMERA_OK;
            if(ok){ int pos=ePos.data.u8[0]; strncpy(ids[pos],id,CAM_ID_MAX); l|=(pos==0); r|=(pos==1); }
            ACameraMetadata_free(m);
        }
        ACameraManager_deleteCameraIdList(list); return l&&r;
    }

    bool initCamera(int w,int h,int fps,int br,int eye){
        char cam[2][CAM_ID_MAX]={{0}}; if(!getIds(cam)){LOGE("passthrough cams not found"); return false;} const char* id=cam[eye];
        static ACameraDevice_StateCallbacks devCb{nullptr,nullptr,nullptr};
        if(ACameraManager_openCamera(cameraMgr_,id,&devCb,&device_[eye])!=ACAMERA_OK){LOGE("openCamera %s",id); return false;}

        // Encoder
        AMediaFormat* fmt=AMediaFormat_new();
        AMediaFormat_setString(fmt,AMEDIAFORMAT_KEY_MIME,"video/avc");
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_WIDTH,w);
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_HEIGHT,h);
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_FRAME_RATE,fps);
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_BIT_RATE,br);
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_I_FRAME_INTERVAL,1);
        AMediaFormat_setInt32(fmt,AMEDIAFORMAT_KEY_COLOR_FORMAT,0x7F000789); // COLOR_FormatSurface
        codec_[eye]=AMediaCodec_createEncoderByType("video/avc");
        if(AMediaCodec_configure(codec_[eye],fmt,nullptr,nullptr,AMEDIACODEC_CONFIGURE_FLAG_ENCODE)!=AMEDIA_OK){LOGE("codec config"); return false;}
        ANativeWindow* win=nullptr; if(AMediaCodec_createInputSurface(codec_[eye],&win)!=AMEDIA_OK||!win){LOGE("inputSurface"); return false;}
        surface_[eye]=win; AMediaCodec_start(codec_[eye]); AMediaFormat_delete(fmt);

        // Capture session → encoder surface
        ACaptureSessionOutputContainer* cont=nullptr; ACaptureSessionOutputContainer_create(&cont);
        ACaptureSessionOutput* out=nullptr; ACaptureSessionOutput_create(surface_[eye],&out); ACaptureSessionOutputContainer_add(cont,out);
        ACameraOutputTarget* tgt=nullptr; ACameraOutputTarget_create(surface_[eye],&tgt);
        ACameraDevice_createCaptureRequest(device_[eye],TEMPLATE_RECORD,&request_[eye]);
        ACaptureRequest_addTarget(request_[eye],tgt);
        ACameraCaptureSession_stateCallbacks scb{nullptr,nullptr,nullptr};
        if(ACameraDevice_createCaptureSession(device_[eye],cont,&scb,&session_[eye])!=ACAMERA_OK) return false;
        ACaptureRequest* reqArr[1] = { request_[eye] };
        int64_t seqId;
        ACameraCaptureSession_setRepeatingRequest(session_[eye], nullptr, 1, reqArr,
                                                  reinterpret_cast<int *>(&seqId));
        ACaptureSessionOutput_free(out); ACaptureSessionOutputContainer_free(cont); ACameraOutputTarget_free(tgt);
        return true;
    }

    // ------------------------------ Encoding loop --------------------------
    void encodeLoop(int eye){ constexpr size_t HDR=sizeof(PacketHdr), MTU=1400; std::vector<uint8_t> pkt(MTU);
        while(running_){
            AMediaCodecBufferInfo info; int idx=AMediaCodec_dequeueOutputBuffer(codec_[eye],&info,5'000);
            if(idx<0) continue;
            size_t sz; uint8_t* data=AMediaCodec_getOutputBuffer(codec_[eye],idx,&sz);
            if(!data){AMediaCodec_releaseOutputBuffer(codec_[eye],idx,false); continue;}
            bool idr=(info.flags&AMEDIACODEC_BUFFER_FLAG_KEY_FRAME)!=0;
            auto t=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-startTime_).count();
            uint32_t fid=frameId_[eye]++;
            size_t off=0; while(off+3<info.size&&data[off]==0&&data[off+1]==0) ++off;
            const uint8_t* nal=data+off; size_t left=info.size-off;
            while(left){ uint16_t sq=seq_[eye]++; size_t pay=std::min(left,MTU-HDR);
                PacketHdr* h=reinterpret_cast<PacketHdr*>(pkt.data()); h->eye=eye; h->flags=(idr?1:0)|((pay==left)?2:0);
                h->seq=htons(sq); h->timestamp=htonl((uint32_t)t); h->frameId=htonl(fid);
                memcpy(pkt.data()+HDR,nal,pay);
                sendto(sock_,pkt.data(),HDR+pay,0,(sockaddr*)&addr_,sizeof(addr_));
                nal+=pay; left-=pay;
            }
            AMediaCodec_releaseOutputBuffer(codec_[eye],idx,false);
        }
    }

    // ------------------------------- Members -------------------------------
    JavaVM* vm_; JNIEnv* envMain_; jobject activity_;
    ACameraManager* cameraMgr_{}; ACameraDevice* device_[2]{}; ACameraCaptureSession* session_[2]{}; ACaptureRequest* request_[2]{};
    AMediaCodec* codec_[2]{}; ANativeWindow* surface_[2]{}; std::thread worker_[2]; std::atomic<bool> running_{false};
    int sock_=-1; sockaddr_in addr_{}; uint32_t frameId_[2]{}; uint16_t seq_[2]{}; std::chrono::steady_clock::time_point startTime_;
};

// ------------------------------ JNI bridge -------------------------------
extern "C" {
JNIEXPORT jlong JNICALL Java_com_yourcompany_streamer_NativeBridge_nativeCreate(JNIEnv* env,jobject thiz){JavaVM* vm;env->GetJavaVM(&vm);return reinterpret_cast<jlong>(PassthroughStreamer::create(vm,thiz));}
JNIEXPORT void JNICALL Java_com_yourcompany_streamer_NativeBridge_nativeStart(JNIEnv* env,jobject,jlong h,jstring ip,jint port){auto* s=reinterpret_cast<PassthroughStreamer*>(h); const char* c=env->GetStringUTFChars(ip,nullptr); s->start(c,(uint16_t)port); env->ReleaseStringUTFChars(ip,c);}
JNIEXPORT void JNICALL Java_com_yourcompany_streamer_NativeBridge_nativeStop(JNIEnv*,jobject,jlong h){reinterpret_cast<PassthroughStreamer*>(h)->stop();}
JNIEXPORT jobject JNICALL Java_com_yourcompany_streamer_NativeBridge_nativeSnapshot(JNIEnv* env,jobject,jlong h,jint eye){return reinterpret_cast<PassthroughStreamer*>(h)->snapshot(env,eye);} }
