//! Wrapper cho mlock (chống swap ra đĩa, đã bàn ở câu trả lời về bảo vệ RAM).

use zeroize::Zeroize;

/// Khoá 1 vùng buffer trong RAM, không cho OS swap ra đĩa.
#[cfg(unix)]
pub fn lock_memory(buf: &[u8]) -> std::io::Result<()> {
    let ret = unsafe { libc::mlock(buf.as_ptr() as *const libc::c_void, buf.len()) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
pub fn unlock_memory(buf: &[u8]) -> std::io::Result<()> {
    let ret = unsafe { libc::munlock(buf.as_ptr() as *const libc::c_void, buf.len()) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Windows: VirtualLock/VirtualUnlock — tương đương mlock/munlock của Unix.
/// KHÔNG còn no-op nữa — bản trước (`Ok(())` rỗng) không thực sự khoá gì cả,
/// đây là lỗi bảo mật im lặng đã sửa.
#[cfg(windows)]
pub fn lock_memory(buf: &[u8]) -> std::io::Result<()> {
    use windows_sys::Win32::System::Memory::VirtualLock;
    let ret = unsafe { VirtualLock(buf.as_ptr() as *mut core::ffi::c_void, buf.len()) };
    if ret != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(windows)]
pub fn unlock_memory(buf: &[u8]) -> std::io::Result<()> {
    use windows_sys::Win32::System::Memory::VirtualUnlock;
    let ret = unsafe { VirtualUnlock(buf.as_ptr() as *mut core::ffi::c_void, buf.len()) };
    if ret != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Xóa sạch buffer — dùng `zeroize` thay vì vòng lặp gán `= 0` tay. Lý do:
/// compiler có thể loại bỏ vòng lặp thường (dead store elimination) nếu thấy
/// buffer không bị đọc lại sau đó — đã lưu ý từ câu trả lời rất sớm về
/// mlock/zeroize. `zeroize` đảm bảo phép ghi KHÔNG bị tối ưu hoá mất.
pub fn wipe(buf: &mut [u8]) {
    buf.zeroize();
}

/// Khóa một slice bất kỳ (không chỉ &[u8]) trong RAM, chống swap ra đĩa.
#[cfg(unix)]
pub fn lock_slice<T>(buf: &[T]) -> std::io::Result<()> {
    let ptr = buf.as_ptr() as *const libc::c_void;
    let len = std::mem::size_of_val(buf);
    let ret = unsafe { libc::mlock(ptr, len) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(windows)]
pub fn lock_slice<T>(buf: &[T]) -> std::io::Result<()> {
    use windows_sys::Win32::System::Memory::VirtualLock;
    let ptr = buf.as_ptr() as *mut core::ffi::c_void;
    let len = std::mem::size_of_val(buf);
    let ret = unsafe { VirtualLock(ptr, len) };
    if ret != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
pub fn unlock_slice<T>(buf: &[T]) -> std::io::Result<()> {
    let ptr = buf.as_ptr() as *const libc::c_void;
    let len = std::mem::size_of_val(buf);
    let ret = unsafe { libc::munlock(ptr, len) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(windows)]
pub fn unlock_slice<T>(buf: &[T]) -> std::io::Result<()> {
    use windows_sys::Win32::System::Memory::VirtualUnlock;
    let ptr = buf.as_ptr() as *mut core::ffi::c_void;
    let len = std::mem::size_of_val(buf);
    let ret = unsafe { VirtualUnlock(ptr, len) };
    if ret != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn wipe_zeroes_buffer() {
        let mut buf = vec![1u8, 2, 3, 4, 5];
        wipe(&mut buf);
        assert_eq!(buf, vec![0u8; 5]);
    }

    #[test]
    fn lock_and_unlock_slice_succeed() {
        let buf = vec![0.0f64; 512]; // kich thuoc tuong duong v_proj thuc te
        assert!(lock_slice(&buf).is_ok());
        assert!(unlock_slice(&buf).is_ok());
    }
}