/* -LICENSE-START-
 ** Copyright (c) 2019 Blackmagic Design
 **
 ** Permission is hereby granted, free of charge, to any person or organization
 ** obtaining a copy of the software and accompanying documentation covered by
 ** this license (the "Software") to use, reproduce, display, distribute,
 ** execute, and transmit the Software, and to prepare derivative works of the
 ** Software, and to permit third-parties to whom the Software is furnished to
 ** do so, all subject to the following:
 **
 ** The copyright notices in the Software and this entire statement, including
 ** the above license grant, this restriction and the following disclaimer,
 ** must be included in all copies of the Software, in whole or in part, and
 ** all derivative works of the Software, unless such copies or derivative
 ** works are solely in the form of machine-executable object code generated by
 ** a source language processor.
 **
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 ** FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 ** SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 ** FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 ** ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 ** DEALINGS IN THE SOFTWARE.
 ** -LICENSE-END-
 */

#pragma once

#include "com_ptr.h"
#include "DeckLinkAPI.h"

class LoopThroughVideoFrame
{
public:
	LoopThroughVideoFrame(const com_ptr<IDeckLinkVideoFrame>& videoFrame):
		m_videoFrame(videoFrame),
		m_videoStreamTime(0),
		m_videoFrameDuration(0),
		m_inputFrameStartHardwareTime(0),
		m_inputFrameArrivedHardwareTime(0),
		m_outputFrameScheduledHardwareTime(0),
		m_outputFrameCompletedHardwareTime(0),
		m_outputFrameCompletionResult(bmdOutputFrameDropped)
	{
	}
	virtual ~LoopThroughVideoFrame(void) = default;
	
	void	setVideoFrame(const com_ptr<IDeckLinkVideoFrame>& videoFrame) { m_videoFrame = videoFrame; }
	void	setVideoStreamTime(const BMDTimeValue time) { m_videoStreamTime = time; }
	void	setVideoFrameDuration(const BMDTimeValue duration) { m_videoFrameDuration = duration; }
	void	setInputFrameStartHardwareTime(const BMDTimeValue time) { m_inputFrameStartHardwareTime = time; }
	void	setInputFrameArrivedHardwareTime(const BMDTimeValue time) { m_inputFrameArrivedHardwareTime = time; }
	void	setOutputFrameScheduledHardwareTime(const BMDTimeValue time) { m_outputFrameScheduledHardwareTime = time; }
	void	setOutputFrameCompletedHardwareTime(const BMDTimeValue time) { m_outputFrameCompletedHardwareTime = time; }
	void	setOutputCompletionResult(const BMDOutputFrameCompletionResult result) { m_outputFrameCompletionResult = result; }

	IDeckLinkVideoFrame*			getVideoFramePtr(void) const { return m_videoFrame.get(); }
	BMDTimeValue					getVideoStreamTime(void) const { return m_videoStreamTime; }
	BMDTimeValue					getVideoFrameDuration(void) const { return m_videoFrameDuration; }
	BMDTimeValue					getInputLatency(void) const { return m_inputFrameArrivedHardwareTime - m_inputFrameStartHardwareTime; }
	BMDTimeValue					getProcessingLatency(void) const { return m_outputFrameScheduledHardwareTime - m_inputFrameArrivedHardwareTime; }
	BMDTimeValue					getOutputLatency(void) const { return m_outputFrameCompletedHardwareTime - m_outputFrameScheduledHardwareTime; }
	BMDOutputFrameCompletionResult	getOutputCompletionResult(void) const { return m_outputFrameCompletionResult; }
	com_ptr<IDeckLinkVideoFrame>	m_videoFrame;
private:

	BMDTimeValue					m_videoStreamTime;
	BMDTimeValue					m_videoFrameDuration;
	
	BMDTimeValue					m_inputFrameStartHardwareTime;
	BMDTimeValue					m_inputFrameArrivedHardwareTime;

	BMDTimeValue					m_outputFrameScheduledHardwareTime;
	BMDTimeValue					m_outputFrameCompletedHardwareTime;
	
	BMDOutputFrameCompletionResult	m_outputFrameCompletionResult;
};